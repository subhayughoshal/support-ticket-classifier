"""
Generates the synthetic support-ticket dataset (data/tickets.csv).

The dataset is SELF-CREATED (no external source). Descriptions are built from
hand-written phrase pools per category, combined with varied openers/closers,
plus a small set of deliberately ambiguous tickets. A few duplicate rows and
missing values are injected on purpose so that the data-cleaning step in
src/preprocessing.py has something real to handle.

Run:  python data/generate_dataset.py
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)

OPENERS = ["", "", "", "Hi team, ", "Hello, ", "Please help: ", "Urgent - ", "Good morning, ",
           "Hi, ", "Need support: ", "Dear support, ", "Quick question - "]
CLOSERS = ["", "", "", "", " Please look into it.", " Kindly resolve this soon.", " Thanks.",
           " This is blocking my work.", " Need help asap.", " It started today.",
           " It was working fine yesterday.", " Please advise."]

CORE = {
    "Login Issue": [
        "I forgot my password and cannot login",
        "Password reset email is not arriving",
        "I am unable to sign in to my account",
        "My account got locked after too many wrong attempts",
        "The OTP for login is not being received on my phone",
        "Login page keeps saying invalid username or password",
        "I cannot log in even after resetting my password",
        "Forgot my username and need to recover it",
        "Two factor authentication code is not working at sign in",
        "I get logged out immediately after entering my credentials",
        "The reset password link has expired",
        "Unable to access my account because the login button does nothing",
        "My login credentials are rejected every time",
        "Sign in with single sign on fails for my account",
    ],
    "Application Error": [
        "The application shows an error while saving data",
        "I get a 500 internal server error when I submit the form",
        "The app crashes whenever I open the dashboard",
        "An unexpected exception message appears when I click save",
        "The system throws a null reference error on the orders page",
        "Data is not getting saved and an error popup appears",
        "The application freezes and then closes by itself",
        "I keep getting an error code 404 when opening my profile page",
        "The upload feature fails with an error message",
        "Something went wrong message appears on every submit",
        "The application crashed while I was entering customer details",
        "Error occurred while updating the record and the changes are lost",
        "The delete button gives a server error",
        "Application stops working with a database connection error",
    ],
    "Report": [
        "Please help me generate my monthly sales report",
        "I need the quarterly revenue report for the finance team",
        "Can you share the attendance report for last month",
        "The exported report is missing some columns",
        "I want a weekly summary report of all closed tickets",
        "How do I download the annual performance report",
        "Need a custom report filtered by region and product",
        "The report shows wrong totals compared to my records",
        "Please schedule the inventory report to be emailed every Monday",
        "Unable to export the report to Excel format",
        "I need a report of all customer orders for this year",
        "Request for a profit and loss report for the last quarter",
        "The dashboard report does not match the data in the summary sheet",
        "Please add a new chart to the monthly management report",
    ],
    "Account Update": [
        "I need to change my registered mobile number",
        "Please update my email address on the account",
        "I want to change the name on my profile",
        "Need to update my billing address",
        "Please change my account type from basic to premium",
        "I want to add a new phone number to my profile",
        "How can I update my company details on the account",
        "Please correct the spelling of my name in my account",
        "I would like to change my notification preferences",
        "Need to link a new email id to my existing account",
        "Please update my shipping address before the next order",
        "I want to delete my old phone number from the account",
        "Request to change the account holder name",
        "Please update my profile picture and contact details",
    ],
    "Performance": [
        "The application is very slow today",
        "Pages are taking too long to load",
        "The system response time has become extremely high",
        "Search results take more than a minute to appear",
        "The dashboard is lagging and hangs for a long time",
        "Application performance has degraded since the last update",
        "Loading the orders screen takes forever",
        "The website is slow during peak hours",
        "Everything is running very sluggish and delayed",
        "The app takes ages to respond when I click any button",
        "File upload is extremely slow compared to before",
        "Page load time is more than thirty seconds",
        "The software is very laggy and unresponsive",
        "It takes a very long time to open the customer list",
    ],
}

# Deliberately ambiguous / hard tickets (label = the main intent of the ticket)
HARD = [
    ("The login page is very slow to load", "Performance"),
    ("Error shown when I try to change my email address", "Account Update"),
    ("Report page is taking too long to open", "Performance"),
    ("I cannot generate the report because I am logged out again and again", "Login Issue"),
    ("The system gives an error when I try to export the sales report", "Application Error"),
    ("Need to update my mobile number but the page is not loading", "Account Update"),
    ("Password change is very slow and times out", "Performance"),
    ("My account details are wrong in the monthly report", "Report"),
    ("Application error on the login screen after I reset my password", "Application Error"),
    ("Please update my email so that I can receive the password reset link", "Account Update"),
    ("Dashboard is slow and also shows an error sometimes", "Performance"),
    ("Unable to sign in and the app shows an error message", "Login Issue"),
    ("Why does the profile page take so long to save my new address", "Performance"),
    ("Report generation crashed with an error", "Application Error"),
    ("I want a report of all users who changed their phone number this month", "Report"),
    ("Cannot access my account and need to update my registered email", "Login Issue"),
    ("The reset password page shows a server error", "Application Error"),
    ("Slow response when downloading the attendance report", "Performance"),
    ("Name on my account is wrong in the exported report", "Account Update"),
    ("Please help, nothing works and I cannot get into the system", "Login Issue"),
]

PRIORITY_WEIGHTS = {
    "Login Issue": {"High": 0.55, "Medium": 0.35, "Low": 0.10},
    "Application Error": {"High": 0.65, "Medium": 0.28, "Low": 0.07},
    "Report": {"High": 0.10, "Medium": 0.50, "Low": 0.40},
    "Account Update": {"High": 0.08, "Medium": 0.42, "Low": 0.50},
    "Performance": {"High": 0.45, "Medium": 0.40, "Low": 0.15},
}
STATUS_WEIGHTS = {"Open": 0.35, "In Progress": 0.25, "Closed": 0.40}


def pick(weights):
    return random.choices(list(weights), weights=list(weights.values()))[0]


rows, seen = [], set()
for category, cores in CORE.items():
    target = 40
    attempts = 0
    while sum(1 for r in rows if r[1] == category) < target and attempts < 2000:
        attempts += 1
        text = random.choice(OPENERS) + random.choice(cores) + random.choice(CLOSERS)
        text = text.strip()
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        rows.append((text, category))

for text, category in HARD:
    rows.append((text, category))

random.shuffle(rows)
records = []
for i, (text, cat) in enumerate(rows, start=1):
    records.append({
        "ticket_id": f"T{i:03d}",
        "ticket_description": text,
        "category": cat,
        "priority": pick(PRIORITY_WEIGHTS[cat]),
        "status": pick(STATUS_WEIGHTS),
    })
df = pd.DataFrame(records)

# --- Inject realistic data-quality problems on purpose ---
# 1) a few exact duplicate descriptions (re-submitted tickets, new ticket ids)
dups = df.sample(6, random_state=1).copy()
dups["ticket_id"] = [f"T{len(df) + k + 1:03d}" for k in range(len(dups))]
df = pd.concat([df, dups], ignore_index=True)
# 2) a few missing values (priority/status), and 2 rows with blank description
df.loc[df.sample(4, random_state=2).index, "priority"] = None
df.loc[df.sample(3, random_state=3).index, "status"] = None
df.loc[df.sample(2, random_state=4).index, "ticket_description"] = None
# 3) messy whitespace / casing in a few descriptions
for idx in df.sample(5, random_state=5).index:
    if isinstance(df.at[idx, "ticket_description"], str):
        df.at[idx, "ticket_description"] = "  " + df.at[idx, "ticket_description"].upper() + "!!  "

out = Path(__file__).parent / "tickets.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
