from typing import List, Optional
import pandas as pd
import requests

class SimpleLLM:
    def __init__(self, api_url: str):
        self.api_url = api_url
        
    def call(self, prompt: str) -> str:
        try:
            headers = {
                # "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            return f"API 请求错误: {str(e)}"

class SimpleDataframeSerializer:
    """简化的DataFrame序列化器"""
    def serialize(self,df: pd.DataFrame) -> str:
        # 获取列名和数据类型的映射
        dtype_dict = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        
        return (
            "列信息: "+f"{', '.join(f'{col}({dtype_dict[col]})' for col in df.columns)}"
            # f"前5行数据:\n"           # f"{df.head().to_dict()}"
        )

class SimpleAgent:
    """简化的Agent实现"""
    def __init__(self, dfs: List[pd.DataFrame], llm: Optional[SimpleLLM] = None):
        self.dfs = dfs
        self.llm = llm or SimpleLLM()
        self.serializer = SimpleDataframeSerializer()
    
    def _generate_prompt(self, query: str) -> str:
        # 序列化所有DataFrames
        dfs_data = []
        for i, df in enumerate(self.dfs):
            df_str = self.serializer.serialize(df)
            # 使用三引号来保持格式
            dfs_data.append(f"""
            dataFrame_{i}: {df_str}""" )
        
        # 构建prompt模板
        prompt = f"""
        Available DataFrames:
        {''.join(dfs_data)}
        
        User Query: {query}
        
        Generate Python function based on the information I provided above.
        
        Ensure the function is syntactically correct and performs the required operations 
        to extract or compute the requested information.
        Available dataframes above is the parameter of function, and the function body should return the result according to the user's query.
        The result type can be "string", "number", "dataframe", or "plot".
        
        Example format:
        def query_function([dataFrame_0, dataFrame_1, ...])->result:
            result = # Write code here #
            return result
        
        Only return the function body between the '#' including 'result =' and without 'return result', do not include the function definition or any other text outside the tags.
        
        """
        # 打印生成的prompt
        print(prompt)
        
        return prompt
        
    def chat(self, query: str) -> str:
        # 1. 生成prompt
        prompt = self._generate_prompt(query)
        
        # 2. 调用LLM
        code = self.llm.call(prompt)
        # 3. 检查代码安全性
        is_safe, message = validate_code_safety(code)
        if not is_safe:
            return f"Code safety check failed: {message}"
        
        # 4. 执行代码
        return self._execute_code(code)
    
    def _execute_code(self, code: str) -> str:
        local_vars = {}

        for i, df in enumerate(self.dfs):
            local_vars[f"dataFrame_{i}"] = df
        print("local_vars: ", local_vars, flush=True)
        try:
            exec(code, {}, local_vars)
            return local_vars.get('result', 'No result found')
        except Exception as e:
            return f"Error executing code: {str(e)}"
        
def validate_code_safety(code: str) -> tuple[bool, str]:
    """
    Check if the code contains potentially dangerous operations.
    
    Args:
        code: The code string to check.
    
    Returns:
        tuple[bool, str]: (Is safe, Error message)
    """
    dangerous_imports = [
        'os', 'subprocess', 'sys', 'shutil', 
        'socket', 'requests', 'urllib',
        'pickle', 'marshal'
    ]
    
    dangerous_functions = [
        'eval', 'exec', 'compile',
        'open', 'write', 'system',
        'remove', 'rmdir', 'unlink'
    ]
    
    for imp in dangerous_imports:
        if f"import {imp}" in code or f"from {imp}" in code:
            return False, f"Detected dangerous module import: {imp}"
            
    for func in dangerous_functions:
        if f"{func}(" in code:
            return False, f"Detected dangerous function call: {func}"
    
    file_keywords = ['file', 'open', 'write', 'read']
    for keyword in file_keywords:
        if keyword in code.lower():
            return False, f"Detected suspicious file operation: {keyword}"
    
    return True, "Code check passed"

# 使用示例
if __name__ == "__main__":
    import pandas as pd
    # 创建示例数据
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
    
    # 初始化Agent
    agent = SimpleAgent([df1, df2])

    # s = "result = dataFrame_0['A'].mean()"
    # result = agent._execute_code(s)
    # print(f"计算结果: {result}")

    # agent._generate_prompt("Calculate the sum of column A in the first dataframe")
    agent._generate_prompt("Calculate the average value of column A")

    # 执行查询
    # result = agent.chat("Calculate the sum of column A in the first dataframe")
    # print(result)

    """
        Available DataFrames:

            dataFrame_0: 列信息: A(int64), B(int64)
            dataFrame_1: 列信息: X(int64), Y(int64)

        User Query: Calculate the average value of column A

        Generate Python function based on the information I provided above.

        Ensure the function is syntactically correct and performs the required operations      
        to extract or compute the requested information.
        Available dataframes above is the parameter of function, and the function body should return the result according to the user's query.
        The result type can be "string", "number", "dataframe", or "plot".

        Example format:
        def query_function([dataFrame_0, dataFrame_1, ...])->result:
            result = # Write code here #
            return result

        Only return the function body between the '#' including 'result =' and without 'return result', do not include the function definition or any other text outside the tags.
    """