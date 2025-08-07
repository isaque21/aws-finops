import boto3
import json
import csv
import time
import logging
import os
from unicodedata import normalize
from botocore.exceptions import ClientError


AWS_REGIONS = ['sa-east-1', 'us-east-1']
AWS_PRICING_REGION = 'us-east-1'
OFFERING_CLASS = 'standard'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


SSM_RESOURCE = boto3.client('ssm')
PRICING_RESOURCE = boto3.client('pricing', region_name=AWS_PRICING_REGION)

# Ajuda a mapear licenseModel corretamente pelo engine
def get_license_model(engine):
    engine_lower = engine.lower()
    
    if 'oracle' in engine_lower:
        return 'License included'
    elif 'sqlserver' in engine_lower or 'sql-server' in engine_lower:
        return 'License included'
    else:
        return 'No license required'

def get_location_name(AWS_REGION):
    try:
        response_location = SSM_RESOURCE.get_parameter(Name='/aws/service/global-infrastructure/regions/'+AWS_REGION+'/longName')
        location_name = response_location['Parameter']['Value']
        location_name = normalize('NFKD', location_name).encode('ASCII','ignore').decode('ASCII')
        return location_name
    except ClientError as e:
        
        logger.error(f"Error getting location name for region {AWS_REGION}: {e}")
        return AWS_REGION

def get_available_pricing_attributes():
    """
    Permite consultar todos os atributos válidos para debug via Pricing API.
    """
    try:
        response = PRICING_RESOURCE.describe_services(ServiceCode='AmazonRDS')
        attribute_names = response['Services'][0]['AttributeNames']
        
        print("> Pricing attribute names:", attribute_names)
    except Exception as e:
        print("Erro ao descrever atributos:", e)


def get_rds_price(region, instanceClass, databaseEngine, deploymentOption, licenseModel='No license required'):
    try:
        deployment_mapping = { 'Single-AZ': 'Single-AZ', 'Multi-AZ': 'Multi-AZ' }
        deployment_filter = deployment_mapping.get(deploymentOption, 'Single-AZ')

        # Ajuste atualizado: engine_mapping fiel ao que aparece na tabela de preço
        engine_mapping = {
            'postgres': 'PostgreSQL',
            'mysql': 'MySQL',
            'mariadb': 'MariaDB',
            'oracle-ee': 'Oracle',
            'oracle-se2': 'Oracle',
            'oracle-se1': 'Oracle',
            'sqlserver-ex': 'SQL Server',
            'sqlserver-web': 'SQL Server',
            'sqlserver-se': 'SQL Server',
            'sqlserver-ee': 'SQL Server',
            'aurora-mysql': 'Aurora MySQL',
            'aurora-postgresql': 'Aurora PostgreSQL',
        }
        if databaseEngine.startswith("aurora"):
            mapped_engine = engine_mapping.get(databaseEngine, databaseEngine.replace("-", " ").title())
        else:
            mapped_engine = engine_mapping.get(databaseEngine, databaseEngine.title())

        filters = [
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region},
            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instanceClass},
            {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': mapped_engine},
            {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': deployment_filter},
        ]
        if licenseModel and licenseModel != 'No license required':
            filters.append({'Type': 'TERM_MATCH', 'Field': 'licenseModel', 'Value': licenseModel})

        
        logger.info(f"Pricing query filters: {filters}")

        paginator = PRICING_RESOURCE.get_paginator('get_products')
        response_iterator = paginator.paginate(
            ServiceCode="AmazonRDS",
            Filters=filters,
            PaginationConfig={'PageSize': 100}
        )

        # Resultados
        priceOnDemand = 'not available'
        noUpfront1yr = 'not available'
        noUpfront3yr = 'not available'
        allUpfront1yr = 'not available'
        allUpfront3yr = 'not available'
        partUpfront1yr = 'not available'
        partUpfront3yr = 'not available'
        partHrsUpfront1yr = 'not available'
        partHrsUpfront3yr = 'not available'
        instanceType = instanceClass
        vcpu = 'not available'
        memory = 'not available'

        found_price = False

        
        for response in response_iterator:
            for priceItem in response["PriceList"]:
                found_price = True
                priceItemJson = json.loads(priceItem)
                product = priceItemJson['product']['attributes']
                instanceType = product.get('instanceType', instanceType)
                vcpu = product.get('vcpu', vcpu)
                memory = product.get('memory', memory)
                for terms in priceItemJson['terms']:
                    if terms == 'OnDemand':
                        for code in priceItemJson['terms'][terms]:
                            for rateCode in priceItemJson['terms'][terms][code]['priceDimensions']:
                                pricePerUnit = priceItemJson['terms'][terms][code]['priceDimensions'][rateCode]['pricePerUnit']['USD']
                                priceOnDemand = '$ ' + str(round(float(pricePerUnit)*730, 2)) # mensalizado
                    elif terms == 'Reserved':
                        for code in priceItemJson['terms'][terms]:
                            termAttributes = priceItemJson['terms'][terms][code]['termAttributes']
                            offeringClass = termAttributes.get('OfferingClass', '')
                            purchaseOption = termAttributes.get('PurchaseOption', '')
                            leaseContractLength = termAttributes.get('LeaseContractLength', '')
                            if offeringClass.lower() == OFFERING_CLASS.lower():
                                for rateCode in priceItemJson['terms'][terms][code]['priceDimensions']:
                                    dimension = priceItemJson['terms'][terms][code]['priceDimensions'][rateCode]
                                    unit = dimension['unit']
                                    pricePerUnit = float(dimension['pricePerUnit']['USD'])
                                    if purchaseOption == 'All Upfront' and unit == 'Quantity':
                                        if leaseContractLength == '1yr':
                                            allUpfront1yr = '$ ' + str(round(pricePerUnit, 2))
                                        elif leaseContractLength == '3yr':
                                            allUpfront3yr = '$ ' + str(round(pricePerUnit, 2))
                                    elif purchaseOption == 'No Upfront' and unit == 'Hrs':
                                        if leaseContractLength == '1yr':
                                            noUpfront1yr = '$ ' + str(round(pricePerUnit*730, 2))
                                        elif leaseContractLength == '3yr':
                                            noUpfront3yr = '$ ' + str(round(pricePerUnit*730, 2))
                                    elif purchaseOption == 'Partial Upfront':
                                        if unit == 'Quantity':
                                            if leaseContractLength == '1yr':
                                                partUpfront1yr = '$ ' + str(round(pricePerUnit, 2))
                                            elif leaseContractLength == '3yr':
                                                partUpfront3yr = '$ ' + str(round(pricePerUnit, 2))
                                        elif unit == 'Hrs':
                                            if leaseContractLength == '1yr':
                                                partHrsUpfront1yr = '$ ' + str(round(pricePerUnit*730, 2))
                                            elif leaseContractLength == '3yr':
                                                partHrsUpfront3yr = '$ ' + str(round(pricePerUnit*730, 2))

        if not found_price:
            logger.warning(f"Nenhum preço encontrado para {instanceClass} ({mapped_engine}) em {region} licenseModel={licenseModel}")

        logger.info(f"Pricing OUT: OnDemand={priceOnDemand}, NoUpfront1yr={noUpfront1yr}, AllUpfront1yr={allUpfront1yr}")
        
        return instanceType, memory, vcpu, priceOnDemand, noUpfront1yr, partUpfront1yr, partHrsUpfront1yr, allUpfront1yr, noUpfront3yr, partUpfront3yr, partHrsUpfront3yr, allUpfront3yr

    except ClientError as e:
        logger.error(f"Error getting price for {instanceClass}: {e}")
    
    except Exception as e:
        logger.error(f"Erro inesperado pegando preço de {instanceClass}: {e}")

    return (instanceClass,'not available','not available','not available','not available','not available','not available','not available','not available','not available','not available','not available')

def calculate_savings(current_price_str, recommended_price_str):
    try:
        if current_price_str == 'not available' or recommended_price_str == 'not available':
            return 'not available', 'not available', 'not available'
        current_price = float(current_price_str.replace('$ ', ''))
        recommended_price = float(recommended_price_str.replace('$ ', ''))
        monthly_savings = current_price - recommended_price
        
        annual_savings = monthly_savings * 12
        if current_price > 0:
            savings_percentage = (monthly_savings / current_price) * 100
        else:
            savings_percentage = 0
        monthly_savings_str = f'$ {round(monthly_savings, 2)}'
        annual_savings_str = f'$ {round(annual_savings, 2)}'
        percentage_str = f'{round(savings_percentage, 1)}%'
        return monthly_savings_str, annual_savings_str, percentage_str
    except Exception as e:
        
        logger.error(f"Erro calculando economia: {e}")
        return 'not available', 'not available', 'not available'


def get_rds_recommendations(db_instance, region, account_id):
    
    logger.info(f"[RDS] Getting RDS recommendations for: {db_instance['DBInstanceIdentifier']} in {region}")
    try:
        compute_optimizer = boto3.client('compute-optimizer', region_name=region)
        db_arn = f"arn:aws:rds:{region}:{account_id}:db:{db_instance['DBInstanceIdentifier']}"
        response = compute_optimizer.get_rds_database_recommendations(
            resourceArns=[db_arn], accountIds=[account_id]
        )
        print(f"Response: {response}")

        finding = 'not available'
        currentInstanceClass = db_instance['DBInstanceClass']
        currentCpuUtilization = 'not available'
        currentMemoryUtilization = 'not available'
        currentDatabaseConnections = 'not available'
        futureInstanceClass = 'not available'
        futureCpuUtilization = 'not available'
        futureMemoryUtilization = 'not available'
        futureDatabaseConnections = 'not available'

        instance_recommendation_engine = db_instance['Engine']
        if response.get('rdsDBRecommendations'):
            recommendation = response['rdsDBRecommendations'][0]
            print(f"Recommendation: {recommendation}")
            finding = recommendation.get('instanceFinding', 'not available')
            currentInstanceClass = recommendation.get('currentDBInstanceClass', db_instance['DBInstanceClass'])
            if 'utilizationMetrics' in recommendation:
                for metric in recommendation['utilizationMetrics']:
                    if metric['name'] == 'CPU' and metric['statistic'] == 'Maximum':
                        currentCpuUtilization = str(round(metric['value'], 2)) + ' %'
                    elif metric['name'] == 'Memory' and metric['statistic'] == 'Maximum':
                        currentMemoryUtilization = str(round(metric['value'], 2)) + ' %'
                    elif metric['name'] == 'DatabaseConnections' and metric['statistic'] == 'Maximum':
                        currentDatabaseConnections = str(round(metric['value'], 2))
            # RECOMENDADO
            # -> sempre pega a PRIMEIRA opção recomendada (pode expandir para melhores cenários oportunamente)
            if 'instanceRecommendationOptions' in recommendation and recommendation['instanceRecommendationOptions']:
                print(f"Options: {recommendation['instanceRecommendationOptions']}")
                futureOption = recommendation['instanceRecommendationOptions'][0]
                futureInstanceClass = futureOption.get('dbInstanceClass', 'not available')
                if 'projectedUtilizationMetrics' in futureOption:
                    for metric in futureOption['projectedUtilizationMetrics']:
                        if metric['name'] == 'CPU':
                            futureCpuUtilization = str(round(metric['value'], 2)) + ' %'
                        elif metric['name'] == 'Memory':
                            futureMemoryUtilization = str(round(metric['value'], 2)) + ' %'
                        elif metric['name'] == 'DatabaseConnections':
                            futureDatabaseConnections = str(round(metric['value'], 2))
                # O engine do recomendado pode mudar em cenários de cross-family (muito raro mas possível futuramente)
                instance_recommendation_engine = futureOption.get('engine', db_instance['Engine'])
        recommendations = [
            db_instance['DBInstanceIdentifier'], db_instance['Engine'], currentInstanceClass,
            currentCpuUtilization, currentMemoryUtilization, currentDatabaseConnections,
            finding, futureInstanceClass, futureCpuUtilization, futureMemoryUtilization, futureDatabaseConnections,
            instance_recommendation_engine # ADICIONADO: engine da recomendação
        ]
        return recommendations
    
    except ClientError as e:
        if 'OptInRequired' in str(e):
            
            logger.warning(f"[WARN] Compute Optimizer not enabled for RDS in region {region}")
            return [db_instance['DBInstanceIdentifier'], db_instance['Engine'], db_instance['DBInstanceClass'], 'Compute Optimizer not enabled', 'Compute Optimizer not enabled', 'Compute Optimizer not enabled',
                'Compute Optimizer not enabled', 'not available', 'not available', 'not available', 'not available', db_instance['Engine']]
        else:
            logger.error(f"Error getting recommendations for {db_instance['DBInstanceIdentifier']}: {e}")
            return [db_instance['DBInstanceIdentifier'], db_instance['Engine'], db_instance['DBInstanceClass'],
                    'not available', 'not available', 'not available', 'not available', 'not available', 'not available', 'not available', 'not available', db_instance['Engine']]

def create_csv_file(data):
    headers = [
        'Account', 'Region', 'DB_Instance_Name', 'DB_Instance_ID', 'Status', 'Engine', 'Engine_Version',
        'Multi_AZ', 'Storage_Type', 'Storage_Size_GB',
        'Current_Instance_Class', 'Current_vCPU', 'Current_Memory', 'Current_Max_CPU_Util', 'Current_Max_Memory_Util', 'Current_Max_DB_Connections',
        'Finding', 'Recommended_Instance_Class', 'Recommended_vCPU', 'Recommended_Memory', 'Recommended_Max_CPU_Util', 'Recommended_Max_Memory_Util', 'Recommended_Max_DB_Connections',
        'Current_OnDemand_Monthly', 'Current_No_Upfront_Monthly_1yr', 'Current_Partial_Upfront_Initial_1yr',
        'Current_Partial_Upfront_Monthly_1yr', 'Current_All_Upfront_1yr', 'Current_No_Upfront_Monthly_3yr',
        'Current_Partial_Upfront_Initial_3yr', 'Current_Partial_Upfront_Monthly_3yr', 'Current_All_Upfront_3yr',
        'Recommended_OnDemand_Monthly', 'Recommended_No_Upfront_Monthly_1yr', 'Recommended_Partial_Upfront_Initial_1yr',
        'Recommended_Partial_Upfront_Monthly_1yr', 'Recommended_All_Upfront_1yr', 'Recommended_No_Upfront_Monthly_3yr',
        'Recommended_Partial_Upfront_Initial_3yr', 'Recommended_Partial_Upfront_Monthly_3yr', 'Recommended_All_Upfront_3yr',
        'OnDemand_Monthly_Savings', 'OnDemand_Annual_Savings', 'OnDemand_Savings_Percentage',
        'No_Upfront_1yr_Monthly_Savings', 'No_Upfront_1yr_Annual_Savings', 'No_Upfront_1yr_Savings_Percentage',
        'All_Upfront_1yr_Annual_Savings', 'All_Upfront_3yr_Annual_Savings'
    ]
    
    ts = int(time.time())
    filename = f'rds_cost_optimization_report_{OFFERING_CLASS}_{ts}.csv'
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
    
    logger.info(f"CSV file created: {filename}")
    print(f"\nRelatório RDS gerado com sucesso: {filename}")
    print(f"Localização: {os.path.abspath(filename)}")
    print(f"Total de registros: {len(data)}")
    return filename


def main():
    print("Iniciando análise de otimização de custos RDS...")
    print(f"Regiões analisadas: {AWS_REGIONS}")
    print(f"Classe de oferta: {OFFERING_CLASS}")
    print(f"Região de preços: {AWS_PRICING_REGION}")

    try:
        sts = boto3.client('sts')
        account_info = sts.get_caller_identity()
        account_id = account_info['Account']
        print(f"Account ID: {account_id}")
    except Exception as e:
        logger.error(f"Error getting account ID: {e}")
        account_id = 'unknown'
    lists = []
    total_instances = 0

    # get_available_pricing_attributes() # (DESCOMENTE para introspecção, debug)

    for region in AWS_REGIONS:
        print(f"\nAnalisando região: {region}")
        try:
            rds = boto3.client('rds', region_name=region)
            response = rds.describe_db_instances()
            region_instances = 0
            if response['DBInstances']:
                for db_instance in response['DBInstances']:
                    db_name = db_instance.get('DBName', db_instance['DBInstanceIdentifier'])
                    db_instance_id = db_instance['DBInstanceIdentifier']
                    db_status = db_instance['DBInstanceStatus']
                    db_engine = db_instance['Engine']
                    db_engine_version = db_instance['EngineVersion']
                    db_instance_class = db_instance['DBInstanceClass']
                    db_multi_az = str(db_instance['MultiAZ'])
                    db_storage_type = db_instance.get('StorageType', 'unknown')
                    db_storage_size = str(db_instance.get('AllocatedStorage', 'unknown'))
                    print(f"  Processando: {db_name} ({db_instance_id}) - {db_instance_class}")
                    location = get_location_name(region)
                    deployment_option = 'Multi-AZ' if db_instance['MultiAZ'] else 'Single-AZ'
                    license_model = get_license_model(db_engine)
                    recommendations = get_rds_recommendations(db_instance, region, account_id)
                    (db_instance_id_rec, engine_rec, current_class, current_cpu_util, current_memory_util, current_db_connections, finding, future_class, future_cpu_util, future_memory_util, future_db_connections, recommended_engine) = recommendations
                    # Preço INSTÂNCIA ATUAL
                    currentTypeInfo, currentMemory, currentVcpu, currentPriceOnDemand, currentNoUpfront1yr, currentPartUpfront1yr, currentPartHrsUpfront1yr, currentAllUpfront1yr, currentNoUpfront3yr, currentPartUpfront3yr, currentPartHrsUpfront3yr, currentAllUpfront3yr = get_rds_price(location, db_instance_class, db_engine, deployment_option, license_model)
                    # Preço INSTÂNCIA RECOMENDADA - usa engine da recomendação
                    if future_class == 'not available' or finding == 'not available' or 'not enabled' in str(current_cpu_util):
                        typeNew=memNew=vcpuNew=priceOnDemandNew=noUpfront1yrNew=partUpfront1yrNew=partHrsUpfront1yrNew=allUpfront1yrNew=noUpfront3yrNew=partUpfront3yrNew=partHrsUpfront3yrNew=allUpfront3yrNew='not available'
                        onDemandMonthlySavings = onDemandAnnualSavings = onDemandSavingsPercentage = 'not available'
                        noUpfront1yrMonthlySavings = noUpfront1yrAnnualSavings = noUpfront1yrSavingsPercentage = 'not available'
                        allUpfront1yrAnnualSavings = allUpfront3yrAnnualSavings = 'not available'
                    else:
                        # O engine pode ser diferente (p.ex. upgrade de Aurora), por segurança use "recommended_engine" se vier
                        recommended_license_model = get_license_model(recommended_engine)
                        typeNew, memNew, vcpuNew, priceOnDemandNew, noUpfront1yrNew, partUpfront1yrNew, partHrsUpfront1yrNew, allUpfront1yrNew, noUpfront3yrNew, partUpfront3yrNew, partHrsUpfront3yrNew, allUpfront3yrNew = get_rds_price(location, future_class, recommended_engine, deployment_option, recommended_license_model)
                        onDemandMonthlySavings, onDemandAnnualSavings, onDemandSavingsPercentage = calculate_savings(currentPriceOnDemand, priceOnDemandNew)
                        noUpfront1yrMonthlySavings, noUpfront1yrAnnualSavings, noUpfront1yrSavingsPercentage = calculate_savings(currentNoUpfront1yr, noUpfront1yrNew)
                        _, allUpfront1yrAnnualSavings, _ = calculate_savings(currentAllUpfront1yr, allUpfront1yrNew)
                        _, allUpfront3yrAnnualSavings, _ = calculate_savings(currentAllUpfront3yr, allUpfront3yrNew)
                    listArrRec = [
                        account_id, region, db_name, db_instance_id, db_status, db_engine, db_engine_version,
                        db_multi_az, db_storage_type, db_storage_size,
                        db_instance_class, currentVcpu, currentMemory, current_cpu_util, current_memory_util, current_db_connections,
                        finding, future_class, vcpuNew, memNew, future_cpu_util, future_memory_util, future_db_connections,
                        currentPriceOnDemand, currentNoUpfront1yr, currentPartUpfront1yr, currentPartHrsUpfront1yr, currentAllUpfront1yr,
                        currentNoUpfront3yr, currentPartUpfront3yr, currentPartHrsUpfront3yr, currentAllUpfront3yr,
                        priceOnDemandNew, noUpfront1yrNew, partUpfront1yrNew, partHrsUpfront1yrNew, allUpfront1yrNew, noUpfront3yrNew, partUpfront3yrNew, partHrsUpfront3yrNew, allUpfront3yrNew,
                        onDemandMonthlySavings, onDemandAnnualSavings, onDemandSavingsPercentage,
                        noUpfront1yrMonthlySavings, noUpfront1yrAnnualSavings, noUpfront1yrSavingsPercentage,
                        allUpfront1yrAnnualSavings, allUpfront3yrAnnualSavings
                    ]
                    lists.append(listArrRec)
                    region_instances += 1
                    total_instances += 1
                    
                    time.sleep(0.15)
            print(f"Total de instâncias RDS na região {region}: {region_instances}")
        
        except ClientError as e:
            logger.error(f"Error processing region {region}: {e}")
            print(f"Erro ao processar região {region}: {e}")

    print(f"\nTotal de instâncias RDS analisadas: {total_instances}")
    if lists:
        filename = create_csv_file(lists)
        print(f"\nAnálise RDS concluída! Verifique o arquivo: {filename}")
        print(f"cat {filename}\nhead -5 {filename}\n")
        print('O relatório RDS inclui:')
        print('   • Informações de instância RDS (engine, versão, Multi-AZ)')
        print('   • Preços atuais (Current_*)')
        print('   • Preços recomendados (Recommended_*)')
        print('   • Cálculos de economia (savings)')
        print('   • Métricas de utilização (CPU, Memory, DB Connections)')
    else:
        print("Nenhuma instância RDS encontrada nas regiões especificadas.")

if __name__ == "__main__":
    main()
