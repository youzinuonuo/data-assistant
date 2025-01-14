import os

def create_mock_loan_data(output_dir: str = "data", filename: str = "loan_data.xlsx"):
    """Create mock loan dataset and save to Excel
    
    Args:
        output_dir: Directory to save the file
        filename: Name of the Excel file
    """
    import pandas as pd
    import numpy as np
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建完整的文件路径
    output_path = os.path.join(output_dir, filename)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'loan_amount': np.random.randint(10000, 500000, n_samples),
        'interest_rate': np.random.uniform(3.0, 15.0, n_samples).round(2),
        'term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'income': np.random.randint(30000, 200000, n_samples),
        'age': np.random.randint(22, 70, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment_years': np.random.randint(0, 30, n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.6, n_samples).round(2),
        'loan_status': np.random.choice(['Approved', 'Rejected'], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # 导出到 Excel
    df.to_excel(output_path, index=False, sheet_name='Loan Data')
    print(f"数据已保存到 {output_path}")
    
    return df

# 使用示例
if __name__ == "__main__":
    # 默认会保存到 ./data/loan_data.xlsx
    df = create_mock_loan_data()
    
    # 或者指定其他目录和文件名
    # df = create_mock_loan_data(output_dir="my_data", filename="loans_2024.xlsx")
    
    print("\n数据集预览:")
    print(df.head())
    print("\n数据集信息:")
    print(df.info())