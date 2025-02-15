def process_dataset(df, columns_to_rename=None, columns_to_drop=None):
    """
    Process a pandas DataFrame by renaming and dropping specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to process
    columns_to_rename : dict, optional
        Dictionary where keys are original column names and values are new column names
        Example: {'old_name': 'new_name', 'old_name2': 'new_name2'}
    columns_to_drop : list, optional
        List of column names to drop from the DataFrame
        Example: ['column1', 'column2']
    
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with renamed and dropped columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    processed_df = df.copy()
    
    # Rename columns if specified
    if columns_to_rename:
        processed_df.rename(columns=columns_to_rename, inplace=True)
    
    # Drop columns if specified
    if columns_to_drop:
        processed_df.drop(columns_to_drop, axis=1, inplace=True)
    
    return processed_df