from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    # 添加常用包列表


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
        
        Generate Python function based on the information provided above.
        The following packages are already imported and available:
        ```python
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        ```
        DO NOT include any import statements in your code.
        
        Requirements:
        1. Function should be named 'query_function'
        2. Use **kwargs to receive parameters
        3. Access dataframes as: dataFrame_0, dataFrame_1, etc. from kwargs
        4. Return types can be: string, number, pandas DataFrame, matplotlib Figure (always use plt.gcf() to return the current figure)
        
        Example format:
        def query_function(**kwargs):
            # Get dataframes from kwargs         
            # Your code here using available packages
            # Example: df0.groupby().agg()
            # Example: plt.figure()

            return result
        Complete the query_function and return the full function as the answer without any imports or additional text.
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

            formatted_code = black.format_str(code, mode=black.Mode())
            
            print("\nBlack formatted code:")
            print(f"{formatted_code}")

            # Basic syntax validation
            tree = ast.parse(formatted_code)
            formatted_code = ast.unparse(tree)
            
            print("\nAST formatted code:")
            print(f"{formatted_code}")
            
            
            return True, formatted_code
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Code formatting error: {str(e)}"

    def _execute_code(self, code: str, kwargs: dict) -> str:
        # Prepare execution environment with imported packages
        global_vars = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns
        }
        local_vars = {}
        
        try:
            # 1. Execute function definition
            exec(code, global_vars, local_vars)
            
            # 2. Get defined function
            query_function = local_vars.get('query_function')
            if not query_function:
                return "query_function not found"
   
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
        
        kwargs = self._generate_kwargs(self.dfs)
        # 5. Execute code
        return self._execute_code(formatted_code, kwargs)
    
    def _generate_kwargs(self, dfs: List[pd.DataFrame]) -> dict:
        return {
            f"dataFrame_{i}": df 
            for i, df in enumerate(dfs)
        }

    def testChat(self, query: str):
        code = """
def query_function(**kwargs):
    # Get the dataframe from kwargs
    df0 = kwargs.get('dataFrame_0')
    
    # Count the occurrences of each value in column 'A'
    value_counts = df0['A'].value_counts()
    
    # Generate a bar plot for the value counts
    plt.figure(figsize=(8, 6))
    sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
    plt.xlabel('Values in A')
    plt.ylabel('Count')
    plt.title('Count of Values in Column A')
    
    # Return the current figure
    return plt.gcf()
        """

        is_safe, message = self._validate_code_safety(code)
        if not is_safe:
            return f"Code safety check failed: {message}"
        
        # 4. Format code
        success, formatted_code = self._format_code(code)
        if not success:
            return formatted_code
        
        #test
        df1 = pd.DataFrame({'A': ['hello', 'hell', 'test', 'test', 'blabla'], 'B': [4, 5, 6, 7, 8]})
        df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
        dfs = [df1, df2]
        kwargs = self._generate_kwargs(dfs)
        # 5. Execute code
        return self._execute_code(formatted_code, kwargs)

# 使用示例
if __name__ == "__main__":
    import pandas as pd
    # 创建示例数据
    df1 = pd.DataFrame({'A': ['hello', 'hell', 'test', 'test', 'blabla'], 'B': [4, 5, 6, 7, 8]})
    df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})

    agent = SimpleAgent([df1, df2])
    prompt = agent._generate_prompt("Counting the data in column a and generating a bar graph")

    generated_code_1 = """
def query_function(**kwargs):
    # Get the dataframe from kwargs
    df0 = kwargs.get('dataFrame_0')
    
    # Count the occurrences of each value in column 'A'
    value_counts = df0['A'].value_counts()
    
    # Generate a bar plot for the value counts
    plt.figure(figsize=(8, 6))
    sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
    plt.xlabel('Values in A')
    plt.ylabel('Count')
    plt.title('Count of Values in Column A')
    
    # Return the current figure
    return plt.gcf()
    """ 


