import pandas as pd
class PromptManager:
    
    def get_initial_prompt(self, query: str, df: pd.DataFrame) -> str:
        """Generate initial prompt for code generation"""
        dfs_data = []
        df_str = self.serialize(df)
        dfs_data.append(f"""
        {df_str}
        """)
        prompt = f"""
DataFrame:
{''.join(dfs_data)}

User Query: {query}

Requirements:
1. Generate Python function based on the information provided above, the function should perform operations on the DataFrame according to the User Query.
2. The following packages are already imported and available:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    ```
DO NOT include any import statements in your code.
3. Function should be named 'query_function' and accept one parameter: 'dataFrame'.
4. Return types can be: string, number, pandas DataFrame, matplotlib figure (always use plt.gcf() to return the figure)
5. DO NOT include any comments, explanations, markdown, provide only the Python function code can be executed directly.
6. Start every line inside function with 4 spaces.
Example format:
def query_function(dataFrame):
    # Your code here using available packages
    # Example: dataFrame.groupby('column_name').sum()
    # Example: plt.figure()
    return result
"""
        return prompt
    
    
    def get_error_prompt(self, df: pd.DataFrame, query: str, error: Exception, 
                        code: str, error_type: str) -> str:
        """Generate error-specific prompt for code regeneration"""
        return f"""
Fix the following Python code that failed with error - {error_type} - {str(error)}

Code to fix:
```python
{code}
```
Requirements:
1. Start every line inside function with 4 spaces
2. Return ONLY the function code without any imports, markdown, comments or explanations.
3. Available packages: pd, np, plt, sns
4. Function name: query_function
5. Parameter: dataFrame
6. Return types: string, number, pandas DataFrame, matplotlib figure (always use plt.gcf() to return the figure)
Example format:
def query_function(dataFrame):
    # Your code here using available packages
    # Example: dataFrame.groupby('column_name').sum()
    # Example: plt.figure()
    return result
        
Original query: {query}
DataFrame:
{self.serialize(df)}
"""

    def serialize(self, df: pd.DataFrame) -> str:
        dtype_dict = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        return (
            "Columns:" + f"{', '.join(f'{col}({dtype_dict[col]})' for col in df.columns)}"
        )
