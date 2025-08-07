import boto3
import json
import csv
import time
import logging
import os
from unicodedata import normalize
from botocore.exceptions import ClientError

# Define AWS Region
AWS_REGIONS = ['sa-east-1', 'us-east-1']
# Define AWS Pricing Region (us-east-1 or ap-south-1)
AWS_PRICING_REGION = 'us-east-1'
# Define reservation type (standard or convertible)
OFFERING_CLASS = 'standard'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get AWS Resources
SSM_RESOURCE = boto3.client('ssm')
PRICING_RESOURCE = boto3.client('pricing', region_name=AWS_PRICING_REGION)

############## GET LOCATION NAME #########################################
def get_location_name(AWS_REGION):
    try:
        # Get the name of the location for the region using AWS Systems Manager Parameter Store
        response_location = SSM_RESOURCE.get_parameter(Name='/aws/service/global-infrastructure/regions/'+AWS_REGION+'/longName')
        location_name = response_location['Parameter']['Value']
        location_name = normalize('NFKD', location_name).encode('ASCII','ignore').decode('ASCII')
        logger.info(f"Location for region {AWS_REGION}: {location_name}")
        return location_name
    except ClientError as e:
        logger.error(f"Error getting location name for region {AWS_REGION}: {e}")
        return AWS_REGION

############## GET THE INSTANCE PRICE TYPES #########################################
def get_price(region, typeEc2, operatingSystem, preInstalledSw):
    try:
        logger.info(f"Getting price for {typeEc2} in {region} with OS: {operatingSystem}")
        
        paginator = PRICING_RESOURCE.get_paginator('get_products')
        response_iterator = paginator.paginate(
            ServiceCode="AmazonEC2",
            Filters=[
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'location',
                    'Value': region
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'capacitystatus',
                    'Value': 'Used'
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'tenancy',
                    'Value': 'Shared'
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'instanceType',
                    'Value': typeEc2
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'preInstalledSw',
                    'Value': preInstalledSw
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'operatingSystem',
                    'Value': operatingSystem
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'licenseModel',
                    'Value': 'No License required'
                }
            ],
            PaginationConfig={
                'PageSize': 100
            }
        )
        
        # Initialize variables with default values
        priceOnDemand = 'not available'
        noUpfront1yr = 'not available'
        noUpfront3yr = 'not available'
        allUpfront1yr = 'not available'
        allUpfront3yr = 'not available'
        partUpfront1yr = 'not available'
        partUpfront3yr = 'not available'
        partHrsUpfront1yr = 'not available'
        partHrsUpfront3yr = 'not available'
        instanceType = typeEc2
        vcpu = 'not available'
        memory = 'not available'
        
        for response in response_iterator:
            for priceItem in response["PriceList"]:
                priceItemJson = json.loads(priceItem)
                instanceType = priceItemJson['product']['attributes']['instanceType']
                vcpu = priceItemJson['product']['attributes']['vcpu']
                memory = priceItemJson['product']['attributes']['memory']
                
                for terms in priceItemJson['terms']:
                    if terms == 'OnDemand':
                        for code in priceItemJson['terms'][terms].keys():
                            for rateCode in priceItemJson['terms'][terms][code]['priceDimensions']:
                                pricePerUnit = priceItemJson['terms'][terms][code]['priceDimensions'][rateCode]['pricePerUnit']['USD']
                                priceOnDemand = '$ ' + str(round(float(pricePerUnit)*730, 2))
                                
                    elif terms == 'Reserved':
                        for code in priceItemJson['terms'][terms].keys():
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
        
        logger.info(f"Pricing retrieved for {instanceType}: OnDemand={priceOnDemand}, NoUpfront1yr={noUpfront1yr}, AllUpfront1yr={allUpfront1yr}")
        return instanceType,memory,vcpu,priceOnDemand,noUpfront1yr,partUpfront1yr,partHrsUpfront1yr,allUpfront1yr,noUpfront3yr,partUpfront3yr,partHrsUpfront3yr,allUpfront3yr
    
    except ClientError as e:
        logger.error(f"Error getting price for {typeEc2}: {e}")
        return typeEc2,'not available','not available','not available','not available','not available','not available','not available','not available','not available','not available','not available'
    except Exception as e:
        logger.error(f"Unexpected error getting price for {typeEc2}: {e}")
        return typeEc2,'not available','not available','not available','not available','not available','not available','not available','not available','not available','not available','not available'

############## CALCULATE SAVINGS ####################################
def calculate_savings(current_price_str, recommended_price_str):
    """Calculate monthly and annual savings between current and recommended pricing"""
    try:
        if current_price_str == 'not available' or recommended_price_str == 'not available':
            return 'not available', 'not available', 'not available'
        
        # Remove currency symbol and convert to float
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
        logger.error(f"Error calculating savings: {e}")
        return 'not available', 'not available', 'not available'

############## GET RECOMMENDATIONS ####################################
def get_recommendations(arn, account, instance_id, platform, region):
    logger.info(f"Getting recommendations for: {instance_id} in {region}")
    
    try:
        compute_optimizer = boto3.client('compute-optimizer', region_name=region)
        response = compute_optimizer.get_ec2_instance_recommendations(
                instanceArns=[
                    arn,
                ],
                accountIds=[
                    account,
                ]
            )
        
        fiding = 'not available'
        oldType = 'not available'
        oldVcpuUtilization = 'not available'
        oldMemoryUtilization = 'not available'
        futureType = 'not available'
        futureVcpuUtilization = 'not available'
        futureMemoryUtilization = 'not available'
        
        if response['instanceRecommendations']:
            fiding = response['instanceRecommendations'][0]['finding']
            oldType = response['instanceRecommendations'][0]['currentInstanceType']
            
            # Get current utilization metrics
            if 'utilizationMetrics' in response['instanceRecommendations'][0]:
                for oldMetric in response['instanceRecommendations'][0]['utilizationMetrics']:
                    if oldMetric['name'] == 'CPU' and oldMetric['statistic'] == 'MAXIMUM':
                        oldVcpuUtilization = str(round(oldMetric['value'], 2)) + ' %'
                    elif oldMetric['name'] == 'MEMORY' and oldMetric['statistic'] == 'MAXIMUM':
                        oldMemoryUtilization = str(round(oldMetric['value'], 2)) + ' %'
            
            # Get future recommendations
            if response['instanceRecommendations'][0]['recommendationOptions']:
                futureType = response['instanceRecommendations'][0]['recommendationOptions'][0]['instanceType']
                if 'projectedUtilizationMetrics' in response['instanceRecommendations'][0]['recommendationOptions'][0]:
                    for metric in response['instanceRecommendations'][0]['recommendationOptions'][0]['projectedUtilizationMetrics']:
                        if metric['name'] == 'CPU':
                            futureVcpuUtilization = str(round(metric['value'], 2)) + ' %'
                        elif metric['name'] == 'MEMORY':
                            futureMemoryUtilization = str(round(metric['value'], 2)) + ' %'
        
        recommendations = [instance_id,platform,oldType,oldMemoryUtilization,oldVcpuUtilization,fiding,futureType,futureMemoryUtilization,futureVcpuUtilization]
        logger.info(f"Recommendations found for {instance_id}: {fiding} -> {futureType}")
        return recommendations
    
    except ClientError as e:
        logger.error(f"Error getting recommendations for {instance_id}: {e}")
        return [instance_id,platform,'not available','not available','not available','not available','not available','not available','not available']

############## CREATE CSV FILE LOCALLY #############################
def create_csv_file(data):
    # Updated headers with both current and recommended pricing
    headers = [
        'Account', 'Region', 'Instance_Name', 'Instance_ID', 'State', 'Platform', 
        'Current_Type', 'Current_Memory', 'Current_Max_Memory_Util', 'Current_vCPU', 'Current_Max_vCPU_Util', 
        'Finding', 'Recommended_Type', 'Recommended_Memory', 'Recommended_Max_Memory_Util', 'Recommended_vCPU', 'Recommended_Max_vCPU_Util',
        
        # Current pricing
        'Current_OnDemand_Monthly', 'Current_No_Upfront_Monthly_1yr', 'Current_Partial_Upfront_Initial_1yr', 
        'Current_Partial_Upfront_Monthly_1yr', 'Current_All_Upfront_1yr', 'Current_No_Upfront_Monthly_3yr', 
        'Current_Partial_Upfront_Initial_3yr', 'Current_Partial_Upfront_Monthly_3yr', 'Current_All_Upfront_3yr',
        
        # Recommended pricing  
        'Recommended_OnDemand_Monthly', 'Recommended_No_Upfront_Monthly_1yr', 'Recommended_Partial_Upfront_Initial_1yr', 
        'Recommended_Partial_Upfront_Monthly_1yr', 'Recommended_All_Upfront_1yr', 'Recommended_No_Upfront_Monthly_3yr', 
        'Recommended_Partial_Upfront_Initial_3yr', 'Recommended_Partial_Upfront_Monthly_3yr', 'Recommended_All_Upfront_3yr',
        
        # Savings calculations
        'OnDemand_Monthly_Savings', 'OnDemand_Annual_Savings', 'OnDemand_Savings_Percentage',
        'No_Upfront_1yr_Monthly_Savings', 'No_Upfront_1yr_Annual_Savings', 'No_Upfront_1yr_Savings_Percentage',
        'All_Upfront_1yr_Annual_Savings', 'All_Upfront_3yr_Annual_Savings'
    ]
    
    # Get current timestamp
    ts = int(time.time())
    filename = f'ec2_cost_optimization_report_{OFFERING_CLASS}_{ts}.csv'
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Write headers to the CSV file
            writer.writerow(headers)
            # Write each row of data to the CSV file
            for row in data:
                writer.writerow(row)
        
        logger.info(f"CSV file created successfully: {filename}")
        print(f"\n✅ Relatório gerado com sucesso: {filename}")
        print(f"📁 Localização: {os.path.abspath(filename)}")
        print(f"📊 Total de registros: {len(data)}")
        return filename
    
    except Exception as e:
        logger.error(f"Error creating CSV file: {e}")
        print(f"❌ Erro ao criar arquivo CSV: {e}")
        return None

############## MAIN FUNCTION #######################################
def main():
    print("🚀 Iniciando análise de otimização de custos EC2...")
    print(f"📊 Regiões analisadas: {AWS_REGIONS}")
    print(f"🏷️  Classe de oferta: {OFFERING_CLASS}")
    print(f"💰 Região de preços: {AWS_PRICING_REGION}")
    
    lists = []
    total_instances = 0
    
    # Iterate over each region
    for region in AWS_REGIONS:
        print(f"\n🔍 Analisando região: {region}")
        
        try:
            # Create an EC2 client for the region
            ec2 = boto3.client('ec2', region_name=region)
            # Get Instances attributes
            instances = ec2.describe_instances()
            
            region_instances = 0
            
            # Check if there are reservations
            if len(instances['Reservations']) > 0:
                for instance in instances['Reservations']:
                    ec2Owner = instance['OwnerId']
                    ec2Az = instance['Instances'][0]['Placement']['AvailabilityZone']
                    ec2Name = '-'
                    if 'Tags' in instance['Instances'][0]:
                        for tag in instance['Instances'][0]['Tags']:
                            if tag['Key'] == 'Name':
                                ec2Name = tag['Value']
                    
                    ec2Id = instance['Instances'][0]['InstanceId']
                    ec2Platform = instance['Instances'][0]['PlatformDetails']
                    ec2State = instance['Instances'][0]['State']['Name']
                    ec2Type = instance['Instances'][0]['InstanceType']
                    
                    print(f"  📋 Processando: {ec2Name} ({ec2Id}) - {ec2Type}")
                    
                    # Get instance type details
                    try:
                        types = ec2.describe_instance_types(InstanceTypes=[ec2Type])
                        for typeEc2 in types['InstanceTypes']:
                            ec2Memory = typeEc2['MemoryInfo']['SizeInMiB']
                            ec2Vcpu = typeEc2['VCpuInfo']['DefaultVCpus']
                        
                        ec2Memory = str(round(ec2Memory/1024, 3)) + ' GiB'
                    except Exception as e:
                        logger.error(f"Error getting instance type details for {ec2Type}: {e}")
                        ec2Memory = 'not available'
                        ec2Vcpu = 'not available'
                    
                    # Determine pre-installed software
                    if 'SQL Server Standard' in ec2Platform:
                        preInstalledSw = 'SQL Std'
                    elif 'SQL Server Enterprise' in ec2Platform:
                        preInstalledSw = 'SQL Ent'
                    elif 'SQL Server Web' in ec2Platform:
                        preInstalledSw = 'SQL Web'
                    else:
                        preInstalledSw = 'NA'
                    
                    # Determine platform
                    if ec2Platform == 'Red Hat Enterprise Linux':
                        platform = 'RHEL'
                    elif ec2Platform == 'Linux/UNIX':
                        platform = 'Linux'
                    elif ec2Platform == 'SUSE Linux':
                        platform = 'SUSE'
                    elif 'Windows' in ec2Platform:
                        platform = 'Windows'
                    else:
                        platform = ec2Platform
                    
                    # Get region location name
                    location = get_location_name(region)
                    
                    # Get recommendations
                    ec2Arn = f'arn:aws:ec2:{region}:{instance["OwnerId"]}:instance/{instance["Instances"][0]["InstanceId"]}'
                    instanceIdFuture, platformFuture, oldType, oldMemoryUtilization, oldVcpuUtilization, fiding, typeFuture, memoryUtilizationFuture, vcpuUtilizationFuture = get_recommendations(ec2Arn, ec2Owner, ec2Id, platform, region)
                    
                    # Get current instance pricing (actual type)
                    print(f"    💰 Obtendo preços atuais para {ec2Type}...")
                    currentTypeInfo, currentMemory, currentVcpu, currentPriceOnDemand, currentNoUpfront1yr, currentPartUpfront1yr, currentPartHrsUpfront1yr, currentAllUpfront1yr, currentNoUpfront3yr, currentPartUpfront3yr, currentPartHrsUpfront3yr, currentAllUpfront3yr = get_price(location, ec2Type, platform, preInstalledSw)
                    
                    # Get recommended instance pricing
                    if typeFuture == 'not available' or fiding == 'not available':
                        print(f"    ⚠️  Sem recomendações disponíveis para {ec2Id}")
                        typeNew=memNew=vcpuNew=priceOnDemandNew=noUpfront1yrNew=partUpfront1yrNew=partHrsUpfront1yrNew=allUpfront1yrNew=noUpfront3yrNew=partUpfront3yrNew=partHrsUpfront3yrNew=allUpfront3yrNew='not available'
                        # Savings calculations (all not available)
                        onDemandMonthlySavings = onDemandAnnualSavings = onDemandSavingsPercentage = 'not available'
                        noUpfront1yrMonthlySavings = noUpfront1yrAnnualSavings = noUpfront1yrSavingsPercentage = 'not available'
                        allUpfront1yrAnnualSavings = allUpfront3yrAnnualSavings = 'not available'
                    else:
                        print(f"    💡 Recomendação: {fiding} -> {typeFuture}")
                        print(f"    💰 Obtendo preços recomendados para {typeFuture}...")
                        typeNew, memNew, vcpuNew, priceOnDemandNew, noUpfront1yrNew, partUpfront1yrNew, partHrsUpfront1yrNew, allUpfront1yrNew, noUpfront3yrNew, partUpfront3yrNew, partHrsUpfront3yrNew, allUpfront3yrNew = get_price(location, typeFuture, platformFuture, preInstalledSw)
                        
                        # Calculate savings
                        print(f"    💵 Calculando economias...")
                        onDemandMonthlySavings, onDemandAnnualSavings, onDemandSavingsPercentage = calculate_savings(currentPriceOnDemand, priceOnDemandNew)
                        noUpfront1yrMonthlySavings, noUpfront1yrAnnualSavings, noUpfront1yrSavingsPercentage = calculate_savings(currentNoUpfront1yr, noUpfront1yrNew)
                        
                        # For all upfront, calculate annual savings only
                        _, allUpfront1yrAnnualSavings, _ = calculate_savings(currentAllUpfront1yr, allUpfront1yrNew)
                        _, allUpfront3yrAnnualSavings, _ = calculate_savings(currentAllUpfront3yr, allUpfront3yrNew)
                        
                        if onDemandMonthlySavings != 'not available':
                            print(f"    💲 Economia On-Demand: {onDemandMonthlySavings}/mês ({onDemandSavingsPercentage})")
                    
                    # Build the complete record with all pricing information
                    listArrRec = [
                        # Basic instance info
                        ec2Owner, ec2Az, ec2Name, instanceIdFuture, ec2State, platformFuture,
                        
                        # Current instance details
                        ec2Type, ec2Memory, oldMemoryUtilization, ec2Vcpu, oldVcpuUtilization,
                        
                        # Recommendation info  
                        fiding, typeNew, memNew, memoryUtilizationFuture, vcpuNew, vcpuUtilizationFuture,
                        
                        # Current pricing
                        currentPriceOnDemand, currentNoUpfront1yr, currentPartUpfront1yr, currentPartHrsUpfront1yr, 
                        currentAllUpfront1yr, currentNoUpfront3yr, currentPartUpfront3yr, currentPartHrsUpfront3yr, currentAllUpfront3yr,
                        
                        # Recommended pricing
                        priceOnDemandNew, noUpfront1yrNew, partUpfront1yrNew, partHrsUpfront1yrNew, 
                        allUpfront1yrNew, noUpfront3yrNew, partUpfront3yrNew, partHrsUpfront3yrNew, allUpfront3yrNew,
                        
                        # Savings calculations
                        onDemandMonthlySavings, onDemandAnnualSavings, onDemandSavingsPercentage,
                        noUpfront1yrMonthlySavings, noUpfront1yrAnnualSavings, noUpfront1yrSavingsPercentage,
                        allUpfront1yrAnnualSavings, allUpfront3yrAnnualSavings
                    ]
                    
                    lists.append(listArrRec)
                    region_instances += 1
                    total_instances += 1
                    
                    # Small delay to avoid API throttling
                    time.sleep(0.2)
            
            print(f"📊 Total de instâncias na região {region}: {region_instances}")
            
        except ClientError as e:
            logger.error(f"Error processing region {region}: {e}")
            print(f"❌ Erro ao processar região {region}: {e}")
    
    print(f"\n📈 Total de instâncias analisadas: {total_instances}")
    
    if lists:
        # Create CSV file locally
        filename = create_csv_file(lists)
        if filename:
            print(f"\n🎉 Análise concluída! Verifique o arquivo: {filename}")
            print("\n📋 Para visualizar o arquivo:")
            print(f"   cat {filename}")
            print(f"   head -5 {filename}  # Ver apenas as primeiras linhas")
        else:
            print("❌ Erro ao gerar o relatório CSV")
    else:
        print("⚠️  Nenhuma instância encontrada nas regiões especificadas")

if __name__ == "__main__":
    main()
