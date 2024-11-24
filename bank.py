from typing import List
import pandas as pd
import json
import re
import os
import matplotlib.pyplot as plt


def dynamic_figsize(x_data, base_width=10, scaling_factor=0.15, label_spacing=0.5, max_width=50, height=6):
    """
    Dynamically calculate figsize based on the length of x_data.

    Args:
        x_data (list or array): Data for the x-axis.
        base_width (float): Minimum width of the figure.
        scaling_factor (float): Additional width per data point.
        max_width (float): Maximum allowed width of the figure.
        height (float): Height of the figure.

    Returns:
        tuple: A tuple representing the figsize (width, height).
    """
    width = min(base_width + len(x_data) * scaling_factor * label_spacing, max_width)
    return (width, height)

def load_settings() -> dict:
    # Load settings from ./settings.json
    with open("settings.json", "r", encoding="utf-8") as file:
        settings = json.load(file)
    return settings

def get_settings_sources(settings: dict) -> List[dict]:
    return settings["sources"]

def get_settings_source_types(settings: dict) -> dict:
    return settings["sourceTypes"]

def load_and_transform_data_from_source_enrich_source(original_source: dict) -> dict:
    # Copy original source to avoid modifying it
    source: dict = original_source.copy()

    if "path" not in source:
        raise ValueError("Source must have a path")
    if "type" not in source:
        raise ValueError("Source must have a type")
    if "modifier" not in source:
        source["modifier"] = 1.0
        print("No modifier found, using default value 1.0")
    if "account" not in source:
        source["account"] = "Default"
        print("No account found, using default value Default")
    
    return source

def load_and_transform_data_from_source_add_additional_columns_from_date(original: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = original.copy()

    if "Date" not in data.columns:
        return data
    
    data["Year"] = data["Date"].dt.year
    data["QuarterNumber"] = data["Date"].dt.quarter
    data["Quarter"] = data["Year"].astype(str) + "-Q" + data["QuarterNumber"].astype(str)
    data["Season"] = (data["Date"].dt.month % 12 + 3) // 3
    data["Season"] = data["Season"].map({1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"})
    data["Month"] = data["Date"].dt.strftime("%Y-%m")
    data["MonthName"] = data["Date"].dt.strftime("%B")
    data["DayOfMonth"] = data["Date"].dt.day
    # data["Week"] = data["Date"].dt.strftime("%A")
    data["Week"] = data["Date"].dt.strftime("%G-W%V")
    data["WeekNumber"] = data["Date"].dt.isocalendar().week
    data["Weekday"] = data["Date"].dt.strftime("%A")
    # Weekend is either Saturday or Sunday
    data["IsWeekend"] = data["Date"].dt.dayofweek.isin([5, 6])
    
    return data

def load_and_transform_data_from_source_add_additional_columns_from_amount(original: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = original.copy()

    if "Amount" not in data.columns:
        return data
    
    data["Type"] = data["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")
    
    return data

def load_and_transform_data_from_source_clean_data(original: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = original.copy()

    # Remove rows with missing values in the...
    
    # ...Date column
    data = data.dropna(subset=["Date"])
    
    # ...Amount column
    data = data.dropna(subset=["Amount"])
    
    # Clean text column
    if "Text" in data.columns:
        data["Text"] = data["Text"].str.strip()
        # Remove any text that follows the format M/24-05-05.
        # M can be any letter. The double-digit numbers can be any numbers as well.
        regex = r"[A-Za-z]?\/\d{2}-\d{2}-\d{2}"
        # use a lambda to replace with re.sub
        data["Text"] = data["Text"].apply(lambda x: re.sub(regex, "", x).strip())
        
        # Lowercase all text
        data["Text"] = data["Text"].str.lower()
    
    return data

def load_and_transform_data_from_source(
    source: dict,
    source_types: dict,
    categorizations: dict
) -> pd.DataFrame:
    # Enrich source
    source = load_and_transform_data_from_source_enrich_source(source)
    skiprows: int = source_types[source['type']]['skipRows'] if 'skipRows' in source_types[source['type']] else 0
    
    # Prepare data
    data: pd.DataFrame = pd.DataFrame()
    
    # Load data from file
    if source['path'].endswith('.csv'):
        data = pd.read_csv(source['path'], skiprows=skiprows)
    elif source['path'].endswith('.xlsx'):
        data = pd.read_excel(source['path'], skiprows=skiprows)
    else:
        raise ValueError("Unsupported file format")
    
    # Rename columns
    source_type: dict = source_types[source['type']]
    data = data.rename(columns=source_type['columns'])
    
    # Drop columns not in the source type
    columns_to_drop = [column for column in data.columns if column not in source_type['columns'].values()]
    data = data.drop(columns=columns_to_drop)

    # Ensure the Date column is in datetime format
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])  
    
    # Add additional columns
    data = load_and_transform_data_from_source_add_additional_columns_from_date(data)
    data = load_and_transform_data_from_source_add_additional_columns_from_amount(data)
    
    # Clean up data
    data = load_and_transform_data_from_source_clean_data(data)
    
    # Apply modifier
    data['Amount'] = data['Amount'] * source['modifier']
    
    # Filter away rows with too high or low amounts,
    # since we don't care about those deviations
    if 'minAmount' in source_types[source['type']]:
        data = data[data['Amount'] >= source_types[source['type']['minAmount']]]
    if 'maxAmount' in source_types[source['type']]:
        data = data[data['Amount'] <= source_types[source['type']['maxAmount']]]
    
    # Apply categories
    # If the row as a Text,
    # then check each Category in the categorizations dictionary.
    # Each category has an array of keywords.
    # If any of those keywords are included inside the Text,
    # then the category is applied to the row. We break after the first match.
    if 'Text' in data.columns:
        # for each row in the data
        for index, row in data.iterrows():
            # for each category in the categorizations dictionary
            for category, keywords in categorizations.items():
                # for each keyword in the keywords array
                for keyword in keywords:
                    # if the keyword is in the row's Text
                    if keyword in row['Text']:
                        # apply the category to the row
                        data.at[index, 'Category'] = category
                        break
                # if Category exists on the row, break
                if 'Category' in row:
                    break
    
    # Add Uncategorized category to rows without a category
    data['Category'] = data['Category'].fillna('Uncategorized')
    
    # For each row where the Category is Investments,
    # set the type to Investment
    data.loc[data['Category'] == 'Investments', 'Type'] = 'Investment'
    
    # For each row where the Category is Transfers,
    # set the type to Transfer
    data.loc[data['Category'] == 'Transfers', 'Type'] = 'Transfer'
    
    # All categories
    categories: List[str] = list(categorizations.keys())
    categories.append('Uncategorized')
    
    # All amount columns
    amount_columns: List[str] = [f"Amount{category}" for category in categories]
    amount_columns.append('Amount')
    
    # For each category, create an Amount<CategoryName> column with the amount,
    # for that category on each row.
    for category in categories:
        # Add the total amount for each category
        data[f"Amount{category}"] = data.apply(lambda x: x['Amount'] if x['Category'] == category else 0, axis=1)
        
    return data


def load_and_transform_data_from_sources(
    sources: List[dict],
    source_types: dict,
    categorizations: dict
) -> pd.DataFrame:
    data: pd.DataFrame = None

    for source in sources:
        # Load and transform data from a single source
        source_data = load_and_transform_data_from_source(source, source_types, categorizations)

        # Append data
        if data is None:
            data = source_data
        else:
            data = pd.concat([data, source_data], ignore_index=True)
    
    # Sort data by date in descending order
    data = data.sort_values("Date", ascending=False)
    
    return data

def aggregate_report(title: str, by_column: str, aggregation_mappings: dict, data: pd.DataFrame, is_daily: bool = False) -> pd.DataFrame:
    # if title includes /, save the part before the / as folder path
    # then save the whole title with / replaced by -
    folder_path = ''
    if '/' in title:
        segments: List[str] = title.split('/')
        # everything but the last segment is the folder path
        folder_path: str = '/'.join(segments[:-1])
        os.makedirs(f"output/{folder_path}", exist_ok=True)
    os.makedirs(f"output", exist_ok=True)
    
    file_name = title.replace('/', '-')
    
    # Create a new mapping without the groupby column
    aggregation_mappings_without_groupby_column = aggregation_mappings.copy()
    aggregation_mappings_without_groupby_column.pop(by_column, None)
    
    # Aggregate
    aggregated: pd.DataFrame = data.groupby(by_column).agg(aggregation_mappings_without_groupby_column).reset_index()
    
    # Top 25 Amounts
    top_25_amounts = aggregated.nlargest(25, 'Amount').sort_values('Amount', ascending=False)
    
    path_to: str = f"output/{folder_path}/{file_name}"
    
    # Save data to CSV
    aggregated.to_csv(f"{path_to}-dataAll.csv", index=False)
    top_25_amounts.to_csv(f"{path_to}-dataTop25.csv", index=False)
    
    # Plot a bar chart
    figsize = dynamic_figsize(aggregated[by_column])
    plt.figure(figsize=figsize)
    plt.bar(aggregated[by_column], aggregated['Amount'])
    plt.xlabel(by_column)
    plt.ylabel('Amount')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{path_to}-diagramBar.png")
    plt.close()

    return aggregated

def generate_reports(data: pd.DataFrame, aggregation_mappings: dict, prefix: str = '') -> None:
    prefix_path: str = f"output/{prefix}"
    os.makedirs(prefix_path, exist_ok=True)

    expenses_by_date: pd.DataFrame = aggregate_report(prefix + 'a-byDate', 'Date', aggregation_mappings, data, is_daily=True)
    expenses_by_weekday: pd.DataFrame = aggregate_report(prefix + 'b-byWeekday', 'Weekday', aggregation_mappings, expenses_by_date)
    expenses_by_week: pd.DataFrame = aggregate_report(prefix + 'c-byWeek', 'Week', aggregation_mappings, expenses_by_date)
    expenses_by_month: pd.DataFrame = aggregate_report(prefix + 'd-byMonth', 'Month', aggregation_mappings, expenses_by_date)
    expenses_by_month_name: pd.DataFrame = aggregate_report(prefix + 'e-byMonthName', 'MonthName', aggregation_mappings, expenses_by_date)
    expenses_by_quarter: pd.DataFrame = aggregate_report(prefix + 'f-byQuarter', 'Quarter', aggregation_mappings, expenses_by_date)
    expenses_by_season: pd.DataFrame = aggregate_report(prefix + 'g-bySeason', 'Season', aggregation_mappings, expenses_by_date)
    expenses_by_year: pd.DataFrame = aggregate_report(prefix + 'h-byYear', 'Year', aggregation_mappings, expenses_by_date)
    expenses_by_category: pd.DataFrame = aggregate_report(prefix + 'i-byCategory', 'Category', aggregation_mappings, expenses_by_date)     

def main() -> None:
    # Load settings
    settings = load_settings()
    
    # Get sources
    sources = get_settings_sources(settings)
    
    # Get source types
    source_types = get_settings_source_types(settings)
    
    # Categorizations
    categorizations = settings["categorizations"]
    
    data: pd.DataFrame = load_and_transform_data_from_sources(sources, source_types, categorizations)
    
    # Save data to CSV
    data.to_csv("output/transactions_all.csv", index=False)
    
    # All categories
    all_categories = data['Category'].unique().tolist()
    all_categories.append('Uncategorized')
    
    # All column names we want to map by
    columns_to_aggregate_by: List[str] = []
    for column_to_aggregate_by in all_categories:
        columns_to_aggregate_by.append(f"Amount{column_to_aggregate_by}")
    columns_to_aggregate_by.append('Amount')
    columns_to_aggregate_by.sort()
    
    # Aggregate report mapping
    # Base it on columns_to_map_by
    aggregation_mappings = {}
    for column in columns_to_aggregate_by:
        aggregation_mappings[column] = 'sum'
    aggregation_mappings['Year'] = 'first'
    aggregation_mappings['Quarter'] = 'first'
    aggregation_mappings['Season'] = 'first'
    aggregation_mappings['Month'] = 'first'
    aggregation_mappings['MonthName'] = 'first'
    aggregation_mappings['Week'] = 'first'
    aggregation_mappings['WeekNumber'] = 'first'
    aggregation_mappings['Weekday'] = 'first'
    aggregation_mappings['IsWeekend'] = 'first'
    aggregation_mappings['DayOfMonth'] = 'first'
    
    # Expenses - All
    expenses = data[data['Type'] == 'Expense']
    
    # Use the absolute value of amount columns for expenses
    for column_to_aggregate_by in columns_to_aggregate_by:
        expenses[f"{column_to_aggregate_by}"] = expenses[f"{column_to_aggregate_by}"].abs()
    
    # Reports for all time
    generate_reports(expenses, aggregation_mappings, prefix='expenses/')
    
    # Reports per year
    years: List[str] = expenses['Year'].unique().tolist()
    for year in years:
        expenses_year = expenses[expenses['Year'] == year]
        generate_reports(expenses_year, aggregation_mappings, prefix=f"expenses/{year}/")
        months = expenses_year['Month'].unique().tolist()
        # for month in months:
        #     expenses_month = expenses_year[expenses_year['Month'] == month]
        #     generate_reports(expenses_month, aggregation_mappings, prefix=f"expenses/{year}/{month}/")
    
    # Incomes
    incomes = data[data['Type'] == 'Income']
    incomes_all_path: str = 'output/incomes/incomes_all.csv'
    os.makedirs(os.path.dirname(incomes_all_path), exist_ok=True)
    incomes.to_csv(incomes_all_path, index=False)
    
    # Investments
    investments = data[data['Category'] == 'Investments']
    investments_all_path: str = 'output/investments/investments_all.csv'
    os.makedirs(os.path.dirname(investments_all_path), exist_ok=True)
    investments.to_csv(investments_all_path, index=False)
    
    # Find all instances of Category=Uncategorized,
    # and save a list of distinct Text values to a file
    uncategorized = data[data['Category'] == 'Uncategorized']
    uncategorized_text = uncategorized['Text'].unique()
    with open("output/uncategorized.txt", "w", encoding="utf-8") as file:
        for text in uncategorized_text:
            file.write(text + "\n")

if __name__ == "__main__":
    main()
