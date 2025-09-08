import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data():
    """Load and clean the sales data"""
    print("=" * 60)
    print("SALES DATA ANALYSIS")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv('sales_data.csv')
    
    # Convert sale_data to datetime
    df['sale_data'] = pd.to_datetime(df['sale_data'])
    
    # Rename column for clarity
    df = df.rename(columns={'sale_data': 'sale_date'})
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    return df

def basic_statistics(df):
    """Display basic statistics about the data"""
    print("BASIC STATISTICS")
    print("-" * 30)
    
    print("Dataset Info:")
    print(df.info())
    print()
    
    print("First 5 rows:")
    print(df.head())
    print()
    
    print("Summary Statistics for Units Sold:")
    print(df['units_sold'].describe())
    print()
    
    print("Missing Values:")
    print(df.isnull().sum())
    print()

def category_analysis(df):
    """Analyze sales by category"""
    print("CATEGORY ANALYSIS")
    print("-" * 30)
    
    # Category distribution
    category_stats = df.groupby('category').agg({
        'units_sold': ['count', 'sum', 'mean', 'std', 'min', 'max']
    }).round(2)
    
    category_stats.columns = ['Product_Count', 'Total_Units', 'Avg_Units', 'Std_Units', 'Min_Units', 'Max_Units']
    print("Category Statistics:")
    print(category_stats)
    print()
    
    # Category performance ranking
    category_performance = df.groupby('category')['units_sold'].sum().sort_values(ascending=False)
    print("Category Performance (Total Units Sold):")
    for i, (category, units) in enumerate(category_performance.items(), 1):
        print(f"{i}. {category}: {units} units")
    print()
    
    return category_stats, category_performance

def time_series_analysis(df):
    """Analyze sales over time"""
    print("TIME SERIES ANALYSIS")
    print("-" * 30)
    
    # Daily sales
    daily_sales = df.groupby('sale_date')['units_sold'].sum().reset_index()
    
    print("Daily Sales Summary:")
    print(f"Date Range: {df['sale_date'].min()} to {df['sale_date'].max()}")
    print(f"Total Days: {len(daily_sales)}")
    print(f"Average Daily Sales: {daily_sales['units_sold'].mean():.2f} units")
    print(f"Best Day: {daily_sales.loc[daily_sales['units_sold'].idxmax(), 'sale_date'].strftime('%Y-%m-%d')} ({daily_sales['units_sold'].max()} units)")
    print(f"Worst Day: {daily_sales.loc[daily_sales['units_sold'].idxmin(), 'sale_date'].strftime('%Y-%m-%d')} ({daily_sales['units_sold'].min()} units)")
    print()
    
    return daily_sales

def product_analysis(df):
    """Analyze individual product performance"""
    print("PRODUCT ANALYSIS")
    print("-" * 30)
    
    # Top performing products
    top_products = df.nlargest(5, 'units_sold')[['product_name', 'category', 'units_sold']]
    print("Top 5 Products by Units Sold:")
    print(top_products)
    print()
    
    # Bottom performing products
    bottom_products = df.nsmallest(5, 'units_sold')[['product_name', 'category', 'units_sold']]
    print("Bottom 5 Products by Units Sold:")
    print(bottom_products)
    print()
    
    # Product performance by category
    print("Best Product in Each Category:")
    best_by_category = df.loc[df.groupby('category')['units_sold'].idxmax()]
    print(best_by_category[['product_name', 'category', 'units_sold']])
    print()

def create_visualizations(df, category_stats, daily_sales):
    """Create various visualizations"""
    print("CREATING VISUALIZATIONS...")
    print("-" * 30)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sales Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Category Distribution (Pie Chart)
    category_counts = df['category'].value_counts()
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Product Distribution by Category')
    
    # 2. Units Sold by Category (Bar Chart)
    category_totals = df.groupby('category')['units_sold'].sum().sort_values(ascending=False)
    axes[0, 1].bar(category_totals.index, category_totals.values, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Total Units Sold by Category')
    axes[0, 1].set_ylabel('Units Sold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(category_totals.values):
        axes[0, 1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 3. Daily Sales Trend (Line Chart)
    axes[1, 0].plot(daily_sales['sale_date'], daily_sales['units_sold'], marker='o', linewidth=2, markersize=6)
    axes[1, 0].set_title('Daily Sales Trend')
    axes[1, 0].set_ylabel('Units Sold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Units Sold Distribution (Histogram)
    axes[1, 1].hist(df['units_sold'], bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribution of Units Sold')
    axes[1, 1].set_xlabel('Units Sold')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(df['units_sold'].mean(), color='red', linestyle='--', label=f'Mean: {df["units_sold"].mean():.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('sales_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("Dashboard saved as 'sales_analysis_dashboard.png'")
    plt.show()

def generate_insights(df, category_stats, daily_sales):
    """Generate business insights"""
    print("BUSINESS INSIGHTS")
    print("-" * 30)
    
    # Calculate key metrics
    total_products = len(df)
    total_units = df['units_sold'].sum()
    avg_units_per_product = df['units_sold'].mean()
    best_category = category_stats['Total_Units'].idxmax()
    worst_category = category_stats['Total_Units'].idxmin()
    
    print(f"📊 Total Products Sold: {total_products}")
    print(f"📦 Total Units Sold: {total_units}")
    print(f"📈 Average Units per Product: {avg_units_per_product:.2f}")
    print(f"🏆 Best Performing Category: {best_category} ({category_stats.loc[best_category, 'Total_Units']} units)")
    print(f"⚠️  Category Needing Attention: {worst_category} ({category_stats.loc[worst_category, 'Total_Units']} units)")
    print()
    
    # Sales trend analysis
    sales_trend = daily_sales['units_sold'].pct_change().mean()
    if sales_trend > 0:
        trend_direction = "increasing"
    else:
        trend_direction = "decreasing"
    
    print(f"📈 Sales Trend: {trend_direction} (avg daily change: {sales_trend:.2%})")
    
    # Product diversity analysis
    category_diversity = len(df['category'].unique())
    print(f"🎯 Product Diversity: {category_diversity} categories represented")
    
    # Consistency analysis
    sales_std = df['units_sold'].std()
    sales_cv = sales_std / avg_units_per_product
    if sales_cv < 0.3:
        consistency = "high"
    elif sales_cv < 0.6:
        consistency = "moderate"
    else:
        consistency = "low"
    
    print(f"📊 Sales Consistency: {consistency} (coefficient of variation: {sales_cv:.2f})")
    print()

def main():
    """Main analysis function"""
    try:
        # Load and clean data
        df = load_and_clean_data()
        
        # Perform analyses
        basic_statistics(df)
        category_stats, category_performance = category_analysis(df)
        daily_sales = time_series_analysis(df)
        product_analysis(df)
        
        # Generate insights
        generate_insights(df, category_stats, daily_sales)
        
        # Create visualizations
        create_visualizations(df, category_stats, daily_sales)
        
        print("=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
