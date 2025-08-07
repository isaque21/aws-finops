# AWS FinOps Scripts

Este diretório contém scripts Python para análise e otimização de custos em recursos AWS EC2 e RDS, utilizando recomendações do AWS Compute Optimizer e informações de preços da AWS. Os relatórios gerados ajudam a identificar oportunidades de economia e otimização de instâncias.

## Scripts disponíveis

### 1. get-ec2-recommendations.py

- **Função:**
  - Analisa todas as instâncias EC2 nas regiões configuradas.
  - Obtém recomendações do AWS Compute Optimizer para tipos de instância mais econômicos.
  - Consulta preços On-Demand e Reserved (1 e 3 anos, All/No/Partial Upfront) para instâncias atuais e recomendadas.
  - Calcula possíveis economias mensais e anuais.
  - Gera um relatório CSV detalhado com todas as informações e sugestões de otimização.

- **Relatório gerado:** `ec2_cost_optimization_report_<classe>_<timestamp>.csv`

### 2. get-rds-recommendations.py

- **Função:**
  - Analisa todas as instâncias RDS nas regiões configuradas.
  - Obtém recomendações do AWS Compute Optimizer para classes de instância mais econômicas.
  - Consulta preços On-Demand e Reserved (1 e 3 anos) para instâncias atuais e recomendadas.
  - Calcula possíveis economias mensais e anuais.
  - Gera um relatório CSV detalhado com todas as informações e sugestões de otimização.

- **Relatório gerado:** `rds_cost_optimization_report_<classe>_<timestamp>.csv`

## Pré-requisitos

- Python 3.7+
- Bibliotecas: `boto3`, `botocore`
- Credenciais AWS configuradas (via AWS CLI, variáveis de ambiente ou arquivo de credenciais)
- Permissões necessárias para acessar EC2, RDS, Compute Optimizer, Pricing e SSM

## Como executar

1. Instale as dependências (se necessário):
   ```powershell
   pip install boto3 botocore
   ```

2. Configure suas credenciais AWS:
   - Via AWS CLI:
     ```powershell
     aws configure
     ```
   - Ou defina as variáveis de ambiente `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` e `AWS_DEFAULT_REGION`.

3. Execute o script desejado:
   - Para EC2:
     ```powershell
     python get-ec2-recommendations.py
     ```
   - Para RDS:
     ```powershell
     python get-rds-recommendations.py
     ```

4. O relatório será gerado no mesmo diretório, com instruções exibidas ao final da execução.

## Observações
- Os scripts analisam as regiões definidas nas variáveis `AWS_REGIONS`.
- O Compute Optimizer deve estar habilitado na conta para recomendações.
- Os relatórios CSV incluem detalhes de instância, preços atuais e recomendados, e cálculos de economia.
- Para grandes ambientes, a execução pode demorar devido à limitação de chamadas na API da AWS.

---

**Autor:** Isaque Pereira Alcantara

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
