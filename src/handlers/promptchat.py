from typing import List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import ast
import black
from .prompt_manager import PromptManager
from matplotlib.figure import Figure

class SimpleLLM:
    DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or self.DEFAULT_API_URL
        
    def call(self, prompt: str) -> str:
        return """
    import pandas as pd
    def query_function(dataFrame):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='education', data=dataFrame)
    plt.title('Loan Data by Education Level')
    plt.xlabel('Education')
    plt.ylabel('Count')
    return plt
        """
        # try:
        #     headers = {
        #         # "Authorization": f"Bearer {self.api_key}",
        #         "Content-Type": "application/json"
        #     }
            
        #     payload = {
        #         "prompt": prompt
        #     }
            
        #     response = requests.post(
        #         self.api_url,
        #         headers=headers,
        #         json=payload,
        #         timeout=30
        #     )
            
        #     response.raise_for_status()
        #     result = response.json()
        #     return result.get("response", "")
            
        # except requests.exceptions.RequestException as e:
        #     return f"API request error: {str(e)}"

class SimpleDataframeSerializer:
    """Simplified DataFrame serializer"""
    def serialize(self, df: pd.DataFrame) -> str:
        # Get mapping of column names and data types
        dtype_dict = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        
        return (
            "Columns: " + f"{', '.join(f'{col}({dtype_dict[col]})' for col in df.columns)}"
        )

class SecurityError(Exception):
    """Custom exception for security validation failures"""
    pass

class SimpleAgent:
    MAX_RETRIES = 5
    
    def __init__(self, dfs: List[pd.DataFrame], llm: Optional[SimpleLLM] = None):
        self.dfs = dfs
        self.llm = llm or SimpleLLM()
        self.prompt_manager = PromptManager(dfs)

        self.DANGEROUS_FUNCTIONS = [
            'eval', 'exec', 'compile',
            'open', 'write', 'system',
            'remove', 'rmdir', 'unlink'
        ]
        
        self.FILE_KEYWORDS = ['file', 'open', 'write', 'read']
        
    def _generate_code(self, query: str, df: pd.DataFrame, is_first_attempt: bool, error: Exception = None, code: str = None) -> str:
        """Generate code with appropriate prompt"""
        if is_first_attempt:
            prompt = self.prompt_manager.get_initial_prompt(query, df)
        else:
            prompt = self.prompt_manager.get_error_prompt(
                df=df,
                query=query,
                error=error,
                code=code,
                error_type=type(error).__name__
            )
        print(f"prompt:\n {prompt}")
        code = self.llm.call(prompt)
        
        if not code:
            raise ValueError("LLM returned empty code")
        return code

    def _validate_return_type(self, result: Any) -> None:
        """
        Validate the return type of executed code
        
        Raises:
            ValueError: If return type is not allowed
        """
        # basic type check
        if isinstance(result, (str, int, float, pd.DataFrame, Figure)):
            return
            
        # numpy numeric type check
        if isinstance(result, (np.integer, np.floating)):
            return
            
        # numpy array check
        if isinstance(result, np.ndarray):
            if result.size == 1:
                result = result[0]
                return
                
        raise ValueError(
            f"""Invalid return type: {type(result).__name__}. 
            Expected: string, number, DataFrame, or matplotlib Figure"""
        )

    def _process_code(self, code: str) -> Any:
        """Process generated code with proper error handling"""
        # 1. Format code
        formatted_code = self._format_code(code)

        # 2. Validate security
        self._validate_code_safety(formatted_code)  
        
        # 2. Execute code   
        result = self._execute_code(formatted_code, self.dfs[0])
        
        # 3. Validate return type
        self._validate_return_type(result)
        
        return result

    # deprecated temporary
    # def _generate_kwargs(self, dfs: List[pd.DataFrame]) -> dict:
    #     return {
    #         f"dataFrame_{i}": df 
    #         for i, df in enumerate(dfs)
    #     }


    def _validate_code_safety(self, code: str) -> None:
        """
        Use AST to check code safety
        Raises:
            SecurityError: If code contains any import statements or dangerous functions
        """
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.DANGEROUS_FUNCTIONS:
                            raise SecurityError(
                                f"Dangerous function call detected, {node.func.id}"
                            )
                    
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            if any(keyword in node.func.attr.lower() for keyword in self.FILE_KEYWORDS):
                                raise SecurityError(
                                    f"Suspicious file operation detected, {node.func.value.id}.{node.func.attr}"
                                )
                        
        except SyntaxError as e:
            raise SecurityError(f"Code parsing failed: {str(e)}")
        except Exception as e:
            if not isinstance(e, SecurityError):
                raise SecurityError(f"Unexpected error in security check: {str(e)}")

    def _format_code(self, code: str) -> str:
        """Format code using black formatter"""
        code = code.strip()
        if "import" in code:
            raise RuntimeError(
                "DO NOT include import statements. "
                "All required packages (pd, np, plt, sns) are pre-imported."
            )
        if not code.startswith("def query_function"):
            raise SyntaxError("Code must start with 'def query_function'")
        try:
            mode = black.FileMode()
            return black.format_str(code, mode=mode)
        except black.InvalidInput as e:
            raise SyntaxError(f"{str(e)}")
        except black.NothingChanged:
            return code
        except Exception as e:
            raise RuntimeError(f"Code formatting failed: {str(e)}")

    def _execute_code(self, code: str, df: pd.DataFrame) -> str:
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
                raise ValueError("query_function not found")
   
            # 4. Call function with parameters
            result = query_function(df)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"{str(e)}")

    def chat(self, query: str) -> str:
        """
        Main chat method with unified error handling
        
        Args:
            query: User's query string
            
        Returns:
            str: Response or error message
        """
        retry_count = 0
        last_code = None
        last_error = None
        
        while retry_count <= self.MAX_RETRIES:
            try:
                # First attempt or regeneration
                is_first_attempt = retry_count == 0
                code = self._generate_code(
                    query=query,
                    df=self.dfs[0],
                    is_first_attempt=is_first_attempt,
                    error=last_error,
                    code=last_code
                )
                result = self._process_code(code)
                return result
                
            except Exception as e:
                retry_count += 1
                last_code = code
                last_error = e
        
        return f"Failed after {self.MAX_RETRIES} attempts: {str(last_error)}"
   
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


