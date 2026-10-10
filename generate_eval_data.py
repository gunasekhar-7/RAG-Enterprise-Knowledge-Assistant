import json
import os

os.makedirs('data/sample_documents', exist_ok=True)

dataset = []
documents = []

# Generate 20 positive queries
topics = [
    ("HR", "What is the policy for carrying over unused vacation days?", "Employees may carry forward a maximum of 5 unused vacation days into the following year, expiring March 31st.", "HR_Policy_2026.md"),
    ("HR", "Who is eligible for the 401k match?", "All full-time employees are eligible for the 401k match after 90 days of employment.", "HR_Policy_2026.md"),
    ("HR", "What is the maternity leave policy?", "Employees are entitled to 12 weeks of paid maternity leave after one year of continuous service.", "HR_Policy_2026.md"),
    ("HR", "Are mental health days separate from sick leave?", "Yes, employees are granted 2 dedicated mental health days per year separate from standard sick leave.", "HR_Policy_2026.md"),
    ("HR", "How many times can I work from home per week?", "The company supports a hybrid model allowing employees to work from home up to 3 days per week.", "HR_Policy_2026.md"),
    
    ("IT Security", "How often must passwords be rotated?", "Passwords must be rotated every 90 days and must not reuse the last 5 passwords.", "IT_Security_SOP.md"),
    ("IT Security", "What happens if I lose my YubiKey?", "Report a lost YubiKey immediately to the IT Helpdesk via phone to revoke access.", "IT_Security_SOP.md"),
    ("IT Security", "Can I install third-party browser extensions?", "Only explicitly whitelisted browser extensions may be installed. Requests must go through IT Security.", "IT_Security_SOP.md"),
    ("IT Security", "Is it allowed to use personal USB drives?", "No, the use of personal USB drives or external storage is strictly prohibited on company endpoints.", "IT_Security_SOP.md"),
    ("IT Security", "What is the VPN auto-disconnect timeout?", "The corporate VPN will automatically disconnect after 12 hours of continuous connection.", "IT_Security_SOP.md"),
    
    ("Travel Policy", "What is the daily per diem for meals in London?", "The daily per diem for meals and incidental expenses in London is $125 USD.", "Travel_Policy.md"),
    ("Travel Policy", "Can I book first-class flights?", "First-class flights are only permitted for international travel exceeding 8 hours of continuous flight time.", "Travel_Policy.md"),
    ("Travel Policy", "How many days after travel must expenses be submitted?", "All travel expenses must be submitted via Concur within 14 days of trip completion.", "Travel_Policy.md"),
    ("Travel Policy", "Are alcohol purchases reimbursable?", "Alcohol is only reimbursable during official client dinners, capped at 2 drinks per person.", "Travel_Policy.md"),
    ("Travel Policy", "Is TSA PreCheck covered by the company?", "The company reimburses the cost of TSA PreCheck or Global Entry for frequent travelers (3+ trips/year).", "Travel_Policy.md"),
    
    ("Infrastructure", "What is the primary region for AWS deployments?", "All primary production workloads must be deployed in the us-east-1 AWS region.", "Infra_SOP.md"),
    ("Infrastructure", "How long are database backups retained?", "Automated database backups are retained for 35 days in S3 Glacier.", "Infra_SOP.md"),
    ("Infrastructure", "What is the SLA for P1 incidents?", "P1 (Critical) incidents have a strict 15-minute response SLA and a 2-hour resolution target.", "Infra_SOP.md"),
    ("Infrastructure", "Who approves emergency production deployments?", "Emergency deployments require explicit approval from the VP of Engineering or the on-call Director.", "Infra_SOP.md"),
    ("Infrastructure", "Which tool is used for infrastructure provisioning?", "Terraform is the exclusive tool for provisioning AWS infrastructure resources.", "Infra_SOP.md")
]

doc_contents = {}

for i, (topic, question, answer, doc_name) in enumerate(topics):
    doc_id = f"doc_{doc_name.split('.')[0].lower()}"
    chunk_id = f"{doc_id}_c{i:04d}"
    
    if doc_name not in doc_contents:
        doc_contents[doc_name] = f"# {topic} Document\n\n"
    
    # Add content to doc
    doc_contents[doc_name] += f"## Topic {i}\n{answer}\n\n"
    
    dataset.append({
        "id": f"eval_{i+1:03d}",
        "question": question,
        "expected_answer": answer,
        "expected_source": doc_name,
        "relevant_chunk_ids": [chunk_id],
        "negative_query": False
    })

# Add 5 negative queries
negative_queries = [
    "What is the company policy on employee relocation to Mars?",
    "How do I expense a private jet for my pet?",
    "What is the SLA for alien invasions?",
    "Who is the CEO of our main competitor?",
    "Can I use company funds to buy cryptocurrencies?"
]

for i, nq in enumerate(negative_queries):
    dataset.append({
        "id": f"eval_neg_{i+1:03d}",
        "question": nq,
        "expected_answer": "Insufficient information",
        "expected_source": None,
        "relevant_chunk_ids": [],
        "negative_query": True
    })

# Write dataset
with open('data/golden_evaluation_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

# Write docs
for doc_name, content in doc_contents.items():
    with open(f"data/sample_documents/{doc_name}", 'w') as f:
        f.write(content)

print("Generated dataset and documents.")
