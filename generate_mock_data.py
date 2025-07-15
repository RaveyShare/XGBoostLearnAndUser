

import pandas as pd
import numpy as np
from faker import Faker
import random
import json
from datetime import datetime, timedelta

# --- Configuration ---
NUM_VENDORS = 20
NUM_PROJECTS = 500
MAX_QUOTES_PER_PROJECT = 3

# Initialize Faker for data generation
fake = Faker('zh_CN') # Use Chinese data where possible

# --- Lists for Categorical Data ---
PROJECT_TYPES = ['软件开发', '系统集成', '硬件采购', '咨询服务', '市场推广']
PROJECT_STATUSES = ['completed', 'in_progress', 'planning', 'cancelled']
TEAM_EXPERIENCE_LEVELS = ['junior', 'mixed', 'senior']
PRIORITIES = ['low', 'medium', 'high']
RISK_LEVELS = ['low', 'medium', 'high']
TECH_STACK_OPTIONS = [
    "Python", "JavaScript", "Java", "C#", "Go", "Ruby", "PHP",
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes"
]

# --- 1. Generate Vendors ---
def generate_vendors(num_vendors):
    print("Generating vendors...")
    vendors = []
    for i in range(1, num_vendors + 1):
        vendors.append({
            'id': i,
            'name': fake.company()
        })
    vendors_df = pd.DataFrame(vendors)
    vendors_df.to_csv('vendors.csv', index=False)
    print(f"Generated {len(vendors_df)} vendors.")
    return vendors_df

# --- 2. Generate Projects ---
def generate_projects(num_projects):
    print("Generating projects...")
    projects = []
    for i in range(1, num_projects + 1):
        start_date = fake.date_between(start_date='-3y', end_date='-1y')
        end_date = start_date + timedelta(days=random.randint(30, 365))
        
        # Make data slightly more realistic
        project_type = random.choice(PROJECT_TYPES)
        tech_stack = random.sample(TECH_STACK_OPTIONS, k=random.randint(2, 6)) if project_type in ['软件开发', '系统集成'] else []
        
        projects.append({
            'id': i,
            'name': f"{fake.word().capitalize()}{fake.word().capitalize()}项目",
            'project_type': project_type,
            'description': fake.paragraph(nb_sentences=3),
            'status': random.choice(PROJECT_STATUSES),
            'start_date': start_date,
            'end_date': end_date,
            'function_points': random.randint(10, 500) if project_type in ['软件开发', '系统集成'] else 0,
            'interface_count': random.randint(1, 50) if project_type == '系统集成' else 0,
            'technology_stack': json.dumps(tech_stack),
            'demand_stability_rating': round(random.uniform(1, 5), 1),
            'team_size': random.randint(2, 20),
            'team_experience_level': random.choice(TEAM_EXPERIENCE_LEVELS),
            'priority': random.choice(PRIORITIES),
            'risk_level': random.choice(RISK_LEVELS)
        })
    projects_df = pd.DataFrame(projects)
    projects_df.to_csv('projects.csv', index=False)
    print(f"Generated {len(projects_df)} projects.")
    return projects_df

# --- 3. Generate Quotes ---
def generate_quotes(projects_df, vendors_df):
    print("Generating quotes...")
    quotes = []
    quote_id_counter = 1
    for _, project in projects_df.iterrows():
        num_quotes = random.randint(1, MAX_QUOTES_PER_PROJECT)
        assigned_vendors = random.sample(list(vendors_df['id']), k=num_quotes)
        
        for vendor_id in assigned_vendors:
            # --- Core Logic for Realistic Pricing ---
            # Base price on complexity factors
            base_hours = (project['function_points'] * 1.5 + 
                          project['interface_count'] * 10 + 
                          project['team_size'] * 20)
            
            # Adjust for risk and experience
            if project['risk_level'] == 'high': base_hours *= 1.2
            if project['team_experience_level'] == 'junior': base_hours *= 1.15
            
            # Add some randomness for vendor differences
            vendor_factor = random.uniform(0.9, 1.3)
            quoted_hours = base_hours * vendor_factor
            
            # Assume an average hourly rate with some variance
            hourly_rate = random.uniform(100, 200) # in some currency unit
            quoted_price = (quoted_hours * hourly_rate) / 10000 # in 10k units
            
            # Simulate negotiation for contract price
            negotiation_factor = random.uniform(0.85, 1.05)
            actual_contract_price = quoted_price * negotiation_factor if project['status'] in ['completed', 'in_progress'] else None
            actual_contract_hours = quoted_hours * negotiation_factor if project['status'] in ['completed', 'in_progress'] else None

            quotes.append({
                'id': quote_id_counter,
                'project_id': project['id'],
                'vendor_id': vendor_id,
                'quoted_hours': round(quoted_hours, 2),
                'quoted_price': round(quoted_price, 2),
                'actual_contract_hours': round(actual_contract_hours, 2) if actual_contract_hours else None,
                'actual_contract_price': round(actual_contract_price, 2) if actual_contract_price else None,
                'contract_date': fake.date_between(start_date=project['start_date'], end_date=project['end_date']) if actual_contract_price else None,
                'payment_terms': random.choice(['Net 30', 'Net 60', '50/50 Split'])
            })
            quote_id_counter += 1
            
    quotes_df = pd.DataFrame(quotes)
    quotes_df.to_csv('quotes.csv', index=False)
    print(f"Generated {len(quotes_df)} quotes.")
    return quotes_df

# --- 4. Generate Project Actuals ---
def generate_project_actuals(quotes_df):
    print("Generating project actuals...")
    # Only generate actuals for projects that have a contract
    completed_quotes = quotes_df.dropna(subset=['actual_contract_price'])
    
    # We need to select one quote per project to be the "winner"
    completed_quotes = completed_quotes.loc[completed_quotes.groupby('project_id')['actual_contract_price'].idxmin()]

    actuals = []
    for _, quote in completed_quotes.iterrows():
        # Simulate execution variance
        execution_factor = random.uniform(0.95, 1.25) # Could be over or under budget/time
        actual_effort_hours = quote['actual_contract_hours'] * execution_factor
        actual_final_cost = quote['actual_contract_price'] * execution_factor
        
        actuals.append({
            'id': len(actuals) + 1,
            'project_id': quote['project_id'],
            'actual_effort_hours': round(actual_effort_hours, 2),
            'actual_final_cost': round(actual_final_cost, 2),
            'delivery_quality_score': round(random.uniform(2.5, 5.0), 1),
            'user_satisfaction_score': round(random.uniform(2.0, 5.0), 1)
        })
        
    actuals_df = pd.DataFrame(actuals)
    actuals_df.to_csv('project_actuals.csv', index=False)
    print(f"Generated {len(actuals_df)} project actuals.")
    return actuals_df

# --- Main Execution ---
if __name__ == '__main__':
    vendors_data = generate_vendors(NUM_VENDORS)
    projects_data = generate_projects(NUM_PROJECTS)
    quotes_data = generate_quotes(projects_data, vendors_data)
    project_actuals_data = generate_project_actuals(quotes_data)
    print("\nMock data generation complete!")
    print("Generated files: vendors.csv, projects.csv, quotes.csv, project_actuals.csv")

