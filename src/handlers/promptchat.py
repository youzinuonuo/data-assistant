from typing import List, Optional, Tuple
import pandas as pd
import requests
import ast
import black


class SimpleLLM:
    DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or self.DEFAULT_API_URL
        
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
            return f"API request error: {str(e)}"

class SimpleDataframeSerializer:
    """Simplified DataFrame serializer"""
    def serialize(self, df: pd.DataFrame) -> str:
        # Get mapping of column names and data types
        dtype_dict = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        
        return (
            "Columns: " + f"{', '.join(f'{col}({dtype_dict[col]})' for col in df.columns)}"
        )

class SimpleAgent:
    def __init__(self, dfs: List[pd.DataFrame], llm: Optional[SimpleLLM] = None):
        self.dfs = dfs
        self.llm = llm or SimpleLLM()
        self.serializer = SimpleDataframeSerializer()
        

        self.DANGEROUS_IMPORTS = [
            'os', 'subprocess', 'sys', 'shutil', 
            'socket', 'requests', 'urllib',
            'pickle', 'marshal'
        ]
        
        self.DANGEROUS_FUNCTIONS = [
            'eval', 'exec', 'compile',
            'open', 'write', 'system',
            'remove', 'rmdir', 'unlink'
        ]
        
        self.FILE_KEYWORDS = ['file', 'open', 'write', 'read']
    def _generate_prompt(self, query: str) -> str:
        # 序列化所有DataFrames
        dfs_data = []
        for i, df in enumerate(self.dfs):
            df_str = self.serializer.serialize(df)
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
        def query_function(**kwargs):
            # kwargs is a dictionary contains dataFrame_0, dataFrame_1, etc.
            # Write python code here
            return result
        You should complete the query_function and return the full function as the answer, make sure the function is complete and don't return any hints other than function.
        """
    
        print("prompt:\n", prompt, flush=True)
        return prompt

    def _validate_code_safety(self, code: str) -> Tuple[bool, str]:
        """
        Check if code contains potentially dangerous operations
        
        Args:
            code: Code string to check
        
        Returns:
            Tuple[bool, str]: (is_safe, error_message)
        """
        # Check dangerous imports
        for imp in self.DANGEROUS_IMPORTS:
            if f"import {imp}" in code or f"from {imp}" in code:
                return False, f"Detected dangerous import: {imp}"
                
        # Check dangerous functions
        for func in self.DANGEROUS_FUNCTIONS:
            if f"{func}(" in code:
                return False, f"Detected dangerous function call: {func}"
        
        # Check file operation keywords
        for keyword in self.FILE_KEYWORDS:
            if keyword in code.lower():
                return False, f"Detected suspicious file operation: {keyword}"
        
        return True, "Code safety check passed"

    def _format_code(self, code: str) -> Tuple[bool, str]:
        """
        Format and validate code
        
        Args:
            code: Code string to format
            
        Returns:
            Tuple[bool, str]: (success, formatted_code_or_error)
        """
        try:
            # Basic syntax validation
            tree = ast.parse(code)
            formatted_code = ast.unparse(tree)
            
            print("\nAST formatted code:")
            print(f"{formatted_code}")
            
            # Format with Black
            formatted_code = black.format_str(formatted_code, mode=black.Mode())
            
            print("\nBlack formatted code:")
            print(f"{formatted_code}")
            
            return True, formatted_code
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Code formatting error: {str(e)}"

    def _execute_code(self, code: str) -> str:
        local_vars = {}
        
        try:
            # 1. Execute function definition
            exec(code, {}, local_vars)
            
            # 2. Get defined function
            query_function = local_vars.get('query_function')
            if not query_function:
                return "query_function not found"
            
            # 3. Prepare parameters
            kwargs = {
                f"dataFrame_{i}": df 
                for i, df in enumerate(self.dfs)
            }
            
            # 4. Call function with parameters
            result = query_function(**kwargs)
            return result
            
        except Exception as e:
            return f"Code execution error: {str(e)}"

    def chat(self, query: str) -> str:
        # 1. Generate prompt
        prompt = self._generate_prompt(query)
        
        # 2. Call LLM
        code = self.llm.call(prompt)
        
        # 3. Check code safety
        is_safe, message = self._validate_code_safety(code)
        if not is_safe:
            return f"Code safety check failed: {message}"
        
        # 4. Format code
        success, formatted_code = self._format_code(code)
        if not success:
            return formatted_code
        
        # 5. Execute code
        return self._execute_code(formatted_code)


# 使用示例
if __name__ == "__main__":
    import pandas as pd
    # 创建示例数据
    df1 = pd.DataFrame({'A': ['hello', 'helloa', 'test'], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})

    agent = SimpleAgent([df1, df2])
    # prompt = agent._generate_prompt("Concatenate the all string in column A and sum the column B and return both result")

    s = """def query_function(**kwargs):
    # Extract dataFrame_0 from kwargs
    dataFrame_0 = kwargs.get('dataFrame_0')
    
    # Concatenate all strings in column A of dataFrame_0
    concatenated_string = ''.join(dataFrame_0['A'].astype(str))
    
    # Sum the values in column B of dataFrame_0
    summed_value = dataFrame_0['B'].sum()
    
    # Return both results as a tuple
    return concatenated_string, summed_value

    """

    success, formatted_code = agent._format_code(s)
    if not success:
        print(f"格式化失败: {formatted_code}")
        exit()
    result = agent._execute_code(formatted_code)
    print(f"计算结果: {result}")
    # text = df1['A'].str.extract('([a-zA-Z]+)')
    # print("\n提取字母后的结果:")
    # result = ' '.join(text[0].dropna().tolist())
    # print("\n最终合并的字符串:")
    # print(result)

    # print("原始DataFrame:")
    # print(df1)
    # print("\n列A的内容:")
    # print(df1['A'])  # 提取A列

    # # 对A列进行正则提取
    # text = df1['A'].str.extract('([a-zA-Z]+)')
    # print("\n提取字母后的结果:")
    # print(text)

    # # 合并结果
    # result = ' '.join(text[0].dropna().tolist())
    # print("\n最终合并的字符串:")
    # print(result)
    # 初始化Agent
    # agent = SimpleAgent([df1, df2])

    # test_code = "text = dataFrame_0['A'].str.extract('(a-zA-Z)+')\n    result = ' '.join(text)\n"
    # result = agent._execute_code(test_code)

    # s = "result = dataFrame_0['A'].mean()"
    # result = agent._execute_code(s)

    # agent._generate_prompt("Calculate the sum of column A in the first dataframe")
    # agent._generate_prompt("Calculate the average value of column A")


    # 执行查询
    # result = agent.chat("Calculate the sum of column A in the first dataframe")
    # print(result)
