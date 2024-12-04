from typing import List, Optional
import pandas as pd


class SimpleLLM:
    # TODO: 替换为实际的LLM调用
    def call(self, prompt: str) -> str:
        return "print('Hello World')"  

class SimpleDataframeSerializer:
    """简化的DataFrame序列化器"""
    def serialize(self,df: pd.DataFrame) -> str:
        # 获取列名和数据类型的映射
        dtype_dict = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        
        return f"""
        列信息:
        {', '.join(f'{col}({dtype_dict[col]})' for col in df.columns)}
        
        前5行数据:
        {df.head().to_dict()}
        """

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
            dfs_data.append(f"DataFrame {i}:\n{df_str}")
        
        # 构建prompt模板
        prompt = f"""
        Available DataFrames:
        {'\n'.join(dfs_data)}
        
        User Query: {query}
        
        Generate Python function based on the information I provided above.
        
        Ensure the function is syntactically correct and performs the required operations 
        to extract or compute the requested information.
        Available dataframes above is the parameter of function, and the function body should return the result according to the user's query.
        The result type can be "string", "number", "dataframe", or "plot".
        
        Example format:
        def query_function([dataframe_0, dataframe_1, ...]):
            # Write code here
            return result
        
        Only return the function body after the '#' without '#' and 'return result', do not include the function definition or any other text outside the tags.
        
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
        try:
            exec(code, {}, self.dfs)

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

    # agent._generate_prompt("Calculate the sum of column A in the first dataframe")
    s = "dataframe_0['A'].sum()"
    result = agent._execute_code(s)
    print(f"计算结果: {result}")

    # 执行查询
    # result = agent.chat("Calculate the sum of column A in the first dataframe")
    # print(result)