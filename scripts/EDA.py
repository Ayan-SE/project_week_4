import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time

def check_promotion_distribution(train_data, test_data, promotion_column='promotion_flag'):
    """
    Checks and compares the distribution of a promotion flag between training and test sets.
    """

    if promotion_column not in train_data.columns or promotion_column not in test_data.columns:
        print(f"Error: Column '{promotion_column}' not found in both datasets.")
        return

    train_counts = train_data[promotion_column].value_counts(normalize=True).sort_index()
    test_counts = test_data[promotion_column].value_counts(normalize=True).sort_index()

    print("Training Set Promotion Distribution:")
    print(train_counts)

    print("\nTest Set Promotion Distribution:")
    print(test_counts)

    # Plotting the distributions
    plt.figure(figsize=(5, 3))

    plt.subplot(1, 2, 1)
    sns.countplot(x=promotion_column, data=train_data)
    plt.title("Training Set Promotion Distribution")

    plt.subplot(1, 2, 2)
    sns.countplot(x=promotion_column, data=test_data)
    plt.title("Test Set Promotion Distribution")

    plt.tight_layout()
    plt.show()

    #plotting normalized distributions for better comparison.
    plt.figure(figsize=(5, 3))

    plt.subplot(1, 2, 1)
    train_counts.plot(kind='bar', color='skyblue')
    plt.title("Training Set Promotion Distribution (Normalized)")
    plt.ylabel("Proportion")

    plt.subplot(1, 2, 2)
    test_counts.plot(kind='bar', color='salmon')
    plt.title("Test Set Promotion Distribution (Normalized)")
    plt.ylabel("Proportion")

    plt.tight_layout()
    plt.show()

def analyze_holiday_sales(data, date_column='Date', sales_column='Sales', holiday_column='StateHoliday'):
    """
    Analyzes and compares sales behavior before, during, and after holidays
    """

    if holiday_column not in data.columns:
        print(f"Error: '{holiday_column}' column not found in DataFrame.")
        return

    if date_column not in data.columns or sales_column not in data.columns:
      print(f"Error: '{date_column}' or '{sales_column}' column not found in DataFrame.")
      return

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=date_column)

    holiday_days = data[data[holiday_column] == 1][date_column].dt.date.unique() # Get unique holiday dates.

    for holiday_date in holiday_days:
        holiday_date = pd.to_datetime(holiday_date)

        # Define time windows
        before_start = holiday_date - pd.Timedelta(days=7)
        after_end = holiday_date + pd.Timedelta(days=7)

        # Filter data for each period
        before_holiday = data[(data[date_column] >= before_start) & (data[date_column] < holiday_date)]
        during_holiday = data[data[date_column].dt.date == holiday_date.date()] #Exact holiday day
        after_holiday = data[(data[date_column] > holiday_date) & (data[date_column] <= after_end)]

        # Calculate average sales for each period
        avg_sales_before = before_holiday[sales_column].mean()
        avg_sales_during = during_holiday[sales_column].mean()
        avg_sales_after = after_holiday[sales_column].mean()

        # Visualization
        plt.figure(figsize=(5, 3))
        plt.bar(['Before', 'During', 'After'], [avg_sales_before, avg_sales_during, avg_sales_after])
        plt.title(f'Sales Behavior Around Holiday {holiday_date.date()}')
        plt.ylabel('Average Sales')
        plt.show()

        # Print detailed information
        print(f"\nSales Analysis for Holiday {holiday_date.date()}:")
        print(f"  Before Holiday ({before_start.date()} to {holiday_date.date()}): Average Sales = {avg_sales_before:.2f}")
        print(f"  During Holiday ({holiday_date.date()}): Average Sales = {avg_sales_during:.2f}")
        print(f"  After Holiday ({holiday_date.date()} to {after_end.date()}): Average Sales = {avg_sales_after:.2f}")
        print("-" * 30)

def analyze_seasonal_purchases(data, date_column='Date', sales_column='Sales', seasonal_periods=None):
    """
    Analyzes seasonal purchase behaviors based on defined seasonal periods.
    """

    if seasonal_periods is None:
        print("Error: seasonal_periods must be provided.")
        return

    if date_column not in data.columns or sales_column not in data.columns:
      print(f"Error: '{date_column}' or '{sales_column}' column not found in DataFrame.")
      return

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=date_column)

    seasonal_sales = {}

    for season, (start_date_str, end_date_str) in seasonal_periods.items():
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        seasonal_data = data[(data[date_column] >= start_date) & (data[date_column] <= end_date)]
        seasonal_sales[season] = seasonal_data[sales_column].mean()

        # Visualization for each season
        plt.figure(figsize=(5, 3))
        sns.lineplot(x=seasonal_data[date_column], y=seasonal_data[sales_column])
        plt.title(f'Sales During {season} ({start_date.date()} to {end_date.date()})')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(f"\nAverage Sales During {season}: {seasonal_sales[season]:.2f}")
        print("-" * 30)

    # Overall seasonal comparison (bar plot)
    plt.figure(figsize=(5, 3))
    plt.bar(seasonal_sales.keys(), seasonal_sales.values())
    plt.title('Comparison of Average Sales Across Seasons')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_sales_customer_correlation(data, sales_column='Sales', customer_column='Customers'):
    """
    Analyzes the correlation between sales and the number of customers.
    """

    if sales_column not in data.columns or customer_column not in data.columns:
        print(f"Error: '{sales_column}' or '{customer_column}' column not found in DataFrame.")
        return

    # Calculate the correlation coefficient
    correlation = data[sales_column].corr(data[customer_column])
    print(f"Correlation between {sales_column} and {customer_column}: {correlation:.2f}")

    # Create a scatter plot
    plt.figure(figsize=(5, 3))
    sns.scatterplot(x=customer_column, y=sales_column, data=data)
    plt.title(f'Scatter Plot: {sales_column} vs. {customer_column}')
    plt.xlabel(customer_column)
    plt.ylabel(sales_column)
    plt.show()

    # Create a heatmap of the correlation matrix (optional)
    correlation_matrix = data[[sales_column, customer_column]].corr()
    plt.figure(figsize=(5, 3))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

def analyze_promo_effects(data, date_column='Date', sales_column='Sales', customer_column='Customers', promo_column='Promo'):
    """
    Analyzes how promotions affect sales and customer behavior.
    """

    if promo_column not in data.columns:
        print(f"Error: '{promo_column}' column not found in DataFrame.")
        return

    if sales_column not in data.columns or customer_column not in data.columns:
        print(f"Error: '{sales_column}' or '{customer_column}' column not found in DataFrame.")
        return

    data[date_column] = pd.to_datetime(data[date_column])

    # 1. Overall Promo Effect on Sales and Customers
    print("\nOverall Promo Effect:")
    print(data.groupby(promo_column)[[sales_column, customer_column]].mean())

    # 2. Visualization: Sales and Customers with/without Promo
    plt.figure(figsize=(5, 3))

    plt.subplot(1, 2, 1)
    sns.barplot(x=promo_column, y=sales_column, data=data)
    plt.title(f'{sales_column} with/without Promo')

    plt.subplot(1, 2, 2)
    sns.barplot(x=promo_column, y=customer_column, data=data)
    plt.title(f'{customer_column} with/without Promo')

    plt.tight_layout()
    plt.show()

    # 3. Time Series Analysis (Optional, if you have enough time-series data)
    if len(data[date_column].unique()) > 10: #check if there is enough date data.
        plt.figure(figsize=(5, 3))

        plt.subplot(1, 2, 1)
        sns.lineplot(x=date_column, y=sales_column, hue=promo_column, data=data)
        plt.title(f'{sales_column} Over Time with Promo')

        plt.subplot(1, 2, 2)
        sns.lineplot(x=date_column, y=customer_column, hue=promo_column, data=data)
        plt.title(f'{customer_column} Over Time with Promo')

        plt.tight_layout()
        plt.show()

    # 4. Analyze existing customer behavior (if you have customer IDs)
    if 'CustomerID' in data.columns:
        print("\nAnalyzing Existing Customer Behavior (if CustomerID is available):")
        # Example: Calculate average sales per customer with/without promo
        customer_sales = data.groupby(['CustomerID', promo_column])[sales_column].mean().reset_index()
        print(customer_sales.groupby(promo_column)[sales_column].mean())

def analyze_promo_effectiveness(data, store_column='Store', sales_column='Sales', customer_column='Customers', promo_column='Promo'):
    """
    Analyzes promo effectiveness by store and suggests potential improvements..
    """

    if store_column not in data.columns or promo_column not in data.columns:
        print(f"Error: '{store_column}' or '{promo_column}' column not found in DataFrame.")
        return

    if sales_column not in data.columns or customer_column not in data.columns:
        print(f"Error: '{sales_column}' or '{customer_column}' column not found in DataFrame.")
        return

    # 1. Promo Effectiveness by Store
    store_promo_analysis = data.groupby([store_column, promo_column])[[sales_column, customer_column]].mean().reset_index()

    # 2. Visualization: Sales and Customers by Store and Promo
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x=store_column, y=sales_column, hue=promo_column, data=store_promo_analysis)
    plt.title(f'{sales_column} by Store and Promo')

    plt.subplot(1, 2, 2)
    sns.barplot(x=store_column, y=customer_column, hue=promo_column, data=store_promo_analysis)
    plt.title(f'{customer_column} by Store and Promo')

    plt.tight_layout()
    plt.show()

    # 3. Identify Stores with Low Promo Impact
    promo_impact = store_promo_analysis.pivot_table(index=store_column, columns=promo_column, values=sales_column)
    if len(promo_impact.columns) == 2: #ensure there is a promo and no promo column.
        promo_impact['Sales_Increase'] = promo_impact.iloc[:, 1] - promo_impact.iloc[:, 0]
        low_impact_stores = promo_impact.sort_values('Sales_Increase').head(3)  # Top 3 stores with lowest promo impact
        print("\nStores with Lowest Promo Impact (Based on Sales):")
        print(low_impact_stores)
    else:
        print("\nCannot calculate promo impact. Ensure that there are two unique values in the promo column")

    # 4. Identify Stores with High Potential
    high_potential_stores = data.groupby(store_column)[[sales_column, customer_column]].mean().sort_values(sales_column, ascending=False).head(3) #Top 3 stores with the highest average sales.
    print("\nStores with Highest Potential (Based on Average Sales):")
    print(high_potential_stores)

def analyze_store_trends(file_path, date_column="Date", open_column="Open", value_column="Customers"):
    """
    Analyzes customer behavior trends based on store opening times.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path, parse_dates=[date_column])
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]  # Remove unnamed columns
        
        # Extract hour and time category
        df["Hour"] = df[date_column].dt.hour
        df["TimeCategory"] = df[open_column].apply(lambda x: "Open" if x == 1 else "Closed")

        # Plot trends
        plt.figure(figsize=(5, 3))
        sns.lineplot(data=df, x="Hour", y=value_column, hue="TimeCategory", marker="o")

        plt.title(f"Customer Trends During Store Opening and Closing Times")
        plt.xlabel("Hour of the Day")
        plt.ylabel(value_column)
        plt.legend(title="Store Status")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error processing file: {e}")
        raise

def analyze_customer_behavior_by_time(data, time_column='Date', customer_column='Customers'):
    """
    Analyzes customer behavior trends during store opening and closing times, extracting time from a datetime column.
    """

    if time_column not in data.columns or customer_column not in data.columns:
        print(f"Error: '{time_column}' or '{customer_column}' column not found in DataFrame.")
        return

    try:
        data['TimeOnly'] = pd.to_datetime(data[time_column]).dt.time  # Extract time only
    except ValueError:
        print(f"Error: Time column '{time_column}' is not in a datetime-compatible format.")
        return
    except TypeError:
        print(f"Error: Time column '{time_column}' is not in a datetime-compatible format.")
        return

    # Convert TimeOnly to seconds since midnight
    data['SecondsSinceMidnight'] = data['TimeOnly'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

    # Group by seconds since midnight and calculate average customer count
    time_customer_analysis = data.groupby('SecondsSinceMidnight')[customer_column].mean().reset_index()

    # Visualization: Customer count over time
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=time_customer_analysis['SecondsSinceMidnight'], y=time_customer_analysis[customer_column]) #Corrected line!
    plt.title('Customer Behavior Over Time')
    plt.xlabel('Time (Seconds Since Midnight)')
    plt.ylabel('Average Customer Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Identify peak and low customer times
    peak_time = time_customer_analysis.loc[time_customer_analysis[customer_column].idxmax()]
    low_time = time_customer_analysis.loc[time_customer_analysis[customer_column].idxmin()]

    peak_seconds = peak_time['SecondsSinceMidnight']
    low_seconds = low_time['SecondsSinceMidnight']

    peak_time_obj = time(peak_seconds // 3600, (peak_seconds % 3600) // 60, peak_seconds % 60)
    low_time_obj = time(low_seconds // 3600, (low_seconds % 3600) // 60, low_seconds % 60)

    print(f"Peak Customer Time: {peak_time_obj} (Average Customers: {peak_time[customer_column]:.2f})")
    print(f"Lowest Customer Time: {low_time_obj} (Average Customers: {low_time[customer_column]:2f})")

    data.drop(['TimeOnly', 'SecondsSinceMidnight'], axis=1, inplace=True)
              
def analyze_weekday_open_stores(data, store_column='Store', weekday_column='DayOfWeek', sales_column='Sales', weekday_values=None, weekend_values=None):
    
    """
    Analyzes which stores are open on all weekdays and how it affects their weekend sales,
    using numeric weekday values.
    """

    if weekday_values is None or weekend_values is None:
        print("Error: weekday_values and weekend_values must be provided.")
        return

    if store_column not in data.columns or weekday_column not in data.columns or sales_column not in data.columns:
        print(f"Error: '{store_column}', '{weekday_column}', or '{sales_column}' column not found in DataFrame.")
        return

    # 1. Identify Stores Open on All Weekdays
    weekday_stores = data[data[weekday_column].isin(weekday_values)].groupby(store_column)[weekday_column].nunique()
    open_all_weekdays = weekday_stores[weekday_stores == len(weekday_values)].index.tolist()

    print("\nStores Open on All Weekdays:")
    print(open_all_weekdays)

    # 2. Analyze Weekend Sales for These Stores
    if open_all_weekdays:
        weekend_sales = data[(data[store_column].isin(open_all_weekdays)) & (data[weekday_column].isin(weekend_values))]
        weekend_avg_sales = weekend_sales.groupby(store_column)[sales_column].mean()

        print("\nAverage Weekend Sales for Stores Open on All Weekdays:")
        print(weekend_avg_sales)
    else:
        print("\nNo stores were found that are open on all weekdays.")

def analyze_assortment_sales(data, assortment_column='Assortment', sales_column='Sales'):
    """
    Analyzes how the assortment type affects sales.
    """

    if assortment_column not in data.columns or sales_column not in data.columns:
        print(f"Error: '{assortment_column}' or '{sales_column}' column not found in DataFrame.")
        return

    # 1. Calculate average sales per assortment type
    assortment_sales = data.groupby(assortment_column)[sales_column].mean().reset_index()

    # 2. Visualize sales by assortment type (bar plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=assortment_column, y=sales_column, data=assortment_sales)
    plt.title(f'Average {sales_column} by {assortment_column}')
    plt.xlabel(assortment_column)
    plt.ylabel(sales_column)
    plt.show()

    # 3. Print average sales per assortment type
    print("\nAverage Sales by Assortment Type:")
    print(assortment_sales)

    # 4. Optional: Box plot for distribution within each assortment
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=assortment_column, y=sales_column, data=data)
    plt.title(f'{sales_column} Distribution by {assortment_column}')
    plt.xlabel(assortment_column)
    plt.ylabel(sales_column)
    plt.show()

def analyze_competitor_distance_sales(data, distance_column='DistanceToCompetitor', sales_column='Sales', city_center_column=None):
    """
    Analyzes how the distance to the next competitor affects sales.
    """

    if distance_column not in data.columns or sales_column not in data.columns:
        print(f"Error: '{distance_column}' or '{sales_column}' column not found in DataFrame.")
        return

    # 1. Calculate the correlation between distance and sales
    correlation = data[distance_column].corr(data[sales_column])
    print(f"Correlation between {distance_column} and {sales_column}: {correlation:.2f}")

    # 2. Visualize the relationship (scatter plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=distance_column, y=sales_column, data=data)
    plt.title(f'{sales_column} vs. {distance_column}')
    plt.xlabel(distance_column)
    plt.ylabel(sales_column)
    plt.show()

    # 3. Analyze distance effect in city centers (if city_center_column is provided)
    if city_center_column:
        if city_center_column not in data.columns:
            print(f"Error: '{city_center_column}' column not found in DataFrame.")
            return

        city_center_data = data[data[city_center_column] == 1]  # Assuming 1 indicates city center
        if not city_center_data.empty:
            city_center_correlation = city_center_data[distance_column].corr(city_center_data[sales_column])
            print(f"\nCorrelation between {distance_column} and {sales_column} in City Centers: {city_center_correlation:.2f}")

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=distance_column, y=sales_column, data=city_center_data)
            plt.title(f'{sales_column} vs. {distance_column} (City Centers)')
            plt.xlabel(distance_column)
            plt.ylabel(sales_column)
            plt.show()
        else:
            print("No data found for stores in city centers.")
    else:
        print("\nCity center analysis skipped. Provide 'city_center_column' to analyze.")

    # 4. Optional: Group by distance ranges and analyze
    data['DistanceRange'] = pd.cut(data[distance_column], bins=5)  # Create distance ranges
    distance_range_sales = data.groupby('DistanceRange')[sales_column].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='DistanceRange', y=sales_column, data=distance_range_sales)
    plt.title(f'Average {sales_column} by Distance Range')
    plt.xlabel('Distance Range')
    plt.ylabel(sales_column)
    plt.xticks(rotation=45)
    plt.show()

def analyze_competitor_impact(data, store_column='StoreID', date_column='Date', distance_column='DistanceToCompetitor', sales_column='Sales'):
    """
    Analyzes how the opening or reopening of new competitors affects stores.
    """

    if store_column not in data.columns or date_column not in data.columns or distance_column not in data.columns or sales_column not in data.columns:
        print(f"Error: One or more of '{store_column}', '{date_column}', '{distance_column}', or '{sales_column}' columns not found in DataFrame.")
        return

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=[store_column, date_column])

    # Identify stores with NA distance that later have values
    na_to_value_stores = []
    for store, store_data in data.groupby(store_column):
        na_indices = store_data[store_data[distance_column].isna()].index
        value_indices = store_data[store_data[distance_column].notna()].index

        if not na_indices.empty and not value_indices.empty and min(na_indices) < min(value_indices):
            na_to_value_stores.append(store)

    print(f"\nStores with NA distance that later have values: {na_to_value_stores}")

    # Analyze sales before and after competitor entry for these stores
    for store in na_to_value_stores:
        store_data = data[data[store_column] == store]
        na_indices = store_data[store_data[distance_column].isna()].index
        value_indices = store_data[store_data[distance_column].notna()].index

        na_end_date = store_data.loc[max(na_indices), date_column]
        value_start_date = store_data.loc[min(value_indices), date_column]

        before_competitor = store_data[store_data[date_column] < value_start_date]
        after_competitor = store_data[store_data[date_column] >= value_start_date]

        before_sales = before_competitor[sales_column].mean()
        after_sales = after_competitor[sales_column].mean()

        print(f"\nSales Analysis for Store {store}:")
        print(f"  Before Competitor (up to {na_end_date.date()}): Average Sales = {before_sales:.2f}")
        print(f"  After Competitor (from {value_start_date.date()}): Average Sales = {after_sales:.2f}")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(before_competitor[date_column], before_competitor[sales_column], label='Before Competitor')
        plt.plot(after_competitor[date_column], after_competitor[sales_column], label='After Competitor')
        plt.axvline(x=value_start_date, color='red', linestyle='--', label='Competitor Entry')
        plt.title(f'Sales Trend for Store {store}')
        plt.xlabel('Date')
        plt.ylabel(sales_column)
        plt.legend()
        plt.show()