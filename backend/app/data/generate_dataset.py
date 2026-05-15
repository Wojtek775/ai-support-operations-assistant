"""
generate_dataset.py — Synthetic Support Ticket Dataset Generator

Generates 100 realistic, AI-ready support tickets for benchmarking
and evaluation of the AI Support & Operations Assistant pipeline.

Usage:
    cd backend
    py -3.12 app/data/generate_dataset.py

Output:
    app/data/generated/tickets_dataset.json   — 100 tickets
    app/data/generated/dataset_summary.json   — distribution stats

Design:
    - Deterministic: random.seed(42) — every run produces identical files
    - 80 regular tickets: 8 per category × 10 categories
    - 20 edge case tickets: realistic messy/difficult real-world scenarios
    - Fields: ticket_id, customer_name, email, message, category,
              priority, sentiment, expected_action, difficulty,
              language, edge_case
"""

import json
import random
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed for full reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "generated"
TICKETS_FILE = OUTPUT_DIR / "tickets_dataset.json"
SUMMARY_FILE = OUTPUT_DIR / "dataset_summary.json"

# ---------------------------------------------------------------------------
# Regular ticket templates — 8 per category (10 categories = 80 tickets)
# Each entry: (message, priority, sentiment, expected_action, difficulty)
# ---------------------------------------------------------------------------

REGULAR_TICKETS = {
    "billing": [
        (
            "I was charged twice for my subscription this month. My invoice number is INV-2024-00892. Please refund the duplicate charge immediately.",
            "high", "frustrated", "escalate_billing", "easy"
        ),
        (
            "Hi, I noticed a $49 charge on my card from your company but I cancelled my subscription three months ago. Can you please explain this and issue a refund?",
            "high", "confused", "refund_review", "easy"
        ),
        (
            "We are a 50-seat enterprise account and our invoice for Q1 is incorrect. We were quoted $2,400/month but were billed $3,100. Please have your billing team contact our CFO.",
            "urgent", "frustrated", "escalate_billing", "medium"
        ),
        (
            "Just noticed my plan was upgraded without my consent and now I'm being charged for the Business tier. I only signed up for Basic. Please downgrade and refund the difference.",
            "medium", "angry", "refund_review", "easy"
        ),
        (
            "Hello, I have a question about my latest invoice. The line item for 'API overage' seems higher than expected. Could you send me a detailed usage breakdown?",
            "low", "neutral", "escalate_billing", "easy"
        ),
        (
            "My credit card was declined during renewal even though the card is valid. I've tried updating it three times now. Please fix this before my account is suspended.",
            "high", "frustrated", "escalate_billing", "medium"
        ),
        (
            "We need to update our billing address for tax purposes. Our company moved offices last month. Invoice address should now be: 100 Innovation Blvd, Suite 400, Austin TX 78701.",
            "low", "neutral", "log_to_product_backlog", "easy"
        ),
        (
            "I signed up for the annual plan but I'm being charged monthly. This is causing accounting issues on our end. Please correct this and apply the annual discount retroactively.",
            "medium", "frustrated", "escalate_billing", "medium"
        ),
    ],

    "technical_issue": [
        (
            "The API is returning 500 errors on all POST requests to /v2/analyze since 14:00 UTC today. This is affecting our production workflow for 300+ users.",
            "urgent", "angry", "escalate_engineering", "medium"
        ),
        (
            "Our webhook integration stopped delivering events about 6 hours ago. We've verified the endpoint is live and accessible. Webhook ID: WH-9284.",
            "high", "frustrated", "escalate_engineering", "medium"
        ),
        (
            "The dashboard export to CSV is producing corrupted files. The first column headers are missing and some rows contain encoding errors (UTF-8 issue suspected).",
            "medium", "neutral", "create_bug_ticket", "medium"
        ),
        (
            "Your mobile app crashes immediately on launch on iOS 17.2. It was working fine yesterday. I've tried reinstalling but the issue persists.",
            "high", "frustrated", "create_bug_ticket", "easy"
        ),
        (
            "The search functionality in our workspace is extremely slow — taking 15-20 seconds to return results. This was fine last week. Team size: 25 users.",
            "medium", "frustrated", "escalate_engineering", "medium"
        ),
        (
            "Hi, quick question — is there a rate limit on the /classify endpoint? We're seeing occasional 429 errors during peak hours and want to understand the limits.",
            "low", "neutral", "log_to_product_backlog", "easy"
        ),
        (
            "The two-factor authentication SMS is not being delivered to our users. We've had 12 support escalations from our own customers today because of this.",
            "urgent", "angry", "escalate_engineering", "hard"
        ),
        (
            "The date filter in the analytics report is broken. When I select 'Last 30 days' it sometimes shows data from 60+ days ago. Reproducible 100% of the time.",
            "medium", "neutral", "create_bug_ticket", "medium"
        ),
    ],

    "account_access": [
        (
            "I've been locked out of my account for 2 days now. The password reset email is not arriving even though my email address is correct. I have an important client demo tomorrow.",
            "urgent", "angry", "manual_account_reset", "easy"
        ),
        (
            "One of our team admins left the company and we need to transfer ownership of the workspace to a new admin. The former admin's email is no longer accessible.",
            "high", "neutral", "manual_account_reset", "medium"
        ),
        (
            "I accidentally deleted my account instead of deleting a project. Is there any way to recover it? I had 3 months of data in there.",
            "high", "frustrated", "manual_account_reset", "hard"
        ),
        (
            "Our SSO integration with Okta stopped working after we updated our Okta configuration last night. All 45 team members cannot log in.",
            "urgent", "angry", "escalate_engineering", "hard"
        ),
        (
            "I keep getting 'Invalid session' errors after about 10 minutes of being logged in, even though I haven't timed out. This is happening on Chrome and Firefox.",
            "medium", "frustrated", "escalate_engineering", "medium"
        ),
        (
            "Please help me add my colleague Sarah Chen (sarah.chen@company.io) to our workspace as an admin. I'm the current owner.",
            "low", "positive", "manual_account_reset", "easy"
        ),
        (
            "My account is showing as 'suspended' but I haven't received any notification about why. I'm fully paid up according to my bank statements.",
            "high", "confused", "manual_account_reset", "medium"
        ),
        (
            "We need to merge two workspaces — one created by our US team and one by our EU team. Both contain important historical data. Is this possible?",
            "medium", "neutral", "escalate_engineering", "hard"
        ),
    ],

    "refund": [
        (
            "I purchased the annual plan but I've decided it's not the right fit for our team. We're only 15 days in and I'd like to request a full refund under your 30-day guarantee.",
            "medium", "neutral", "refund_review", "easy"
        ),
        (
            "I was charged for a seat we deleted 2 weeks ago. Your system should have automatically removed the charge. I need a prorated refund for the unused period.",
            "medium", "frustrated", "refund_review", "easy"
        ),
        (
            "We cancelled our enterprise contract last quarter and were promised a refund of $4,800 for the unused months. It's been 6 weeks and we've not received anything.",
            "high", "angry", "escalate_billing", "medium"
        ),
        (
            "I was accidentally signed up for the wrong plan by your sales team. They promised the Starter plan but I was enrolled in Professional. Please refund the difference.",
            "medium", "frustrated", "refund_review", "medium"
        ),
        (
            "Hi, can you confirm your refund policy for annual subscriptions? I'm considering cancelling and want to understand what I'd get back.",
            "low", "neutral", "log_to_product_backlog", "easy"
        ),
        (
            "I received a refund last week but it was for the wrong amount — $150 instead of $300. The original charge was for two seats but you only refunded one.",
            "medium", "frustrated", "refund_review", "medium"
        ),
        (
            "Due to our company downsizing, we need to reduce from 50 seats to 10 seats effective immediately. We'd like a prorated refund for the 40 removed seats.",
            "high", "neutral", "refund_review", "medium"
        ),
        (
            "My payment went through twice due to a browser error on your checkout page. I have screenshots showing both charges. Please refund the duplicate.",
            "high", "frustrated", "refund_review", "easy"
        ),
    ],

    "feature_request": [
        (
            "Would love to see a Slack integration where ticket updates are automatically posted to our #support channel. This would save our team significant time daily.",
            "low", "positive", "log_to_product_backlog", "easy"
        ),
        (
            "Please add bulk export functionality to the analytics dashboard. Right now we can only export page by page which is extremely time-consuming for monthly reports.",
            "medium", "neutral", "log_to_product_backlog", "easy"
        ),
        (
            "As a power user, I'd love keyboard shortcuts throughout the app. Even basic ones like Cmd+K for search or Cmd+N for new ticket would be a huge productivity boost.",
            "low", "positive", "log_to_product_backlog", "easy"
        ),
        (
            "We need a GDPR-compliant data retention policy feature — the ability to set automatic deletion of records older than X days. This is becoming a compliance requirement for us.",
            "medium", "neutral", "log_to_product_backlog", "medium"
        ),
        (
            "It would be really helpful to have conditional logic in your form builder. For example: if category = billing, show the 'invoice number' field.",
            "low", "positive", "log_to_product_backlog", "easy"
        ),
        (
            "Please consider adding a dark mode to the web app. Many of my team members work late and the current bright UI causes eye strain during evening hours.",
            "low", "neutral", "log_to_product_backlog", "easy"
        ),
        (
            "We'd like to see an audit log feature showing who made what changes and when across our workspace. This is critical for our SOC2 compliance requirements.",
            "medium", "neutral", "log_to_product_backlog", "medium"
        ),
        (
            "Is there a roadmap for adding AI-powered auto-routing of tickets based on content? We process 500+ tickets per day and manual routing is a bottleneck.",
            "medium", "positive", "log_to_product_backlog", "medium"
        ),
    ],

    "outage": [
        (
            "Your entire platform appears to be down. We cannot access the dashboard, API calls are timing out, and our entire support workflow has stopped. This has been going on for 20 minutes.",
            "urgent", "angry", "escalate_infrastructure", "easy"
        ),
        (
            "Experiencing intermittent 503 errors on the API. About 40% of requests are failing. This started around 09:15 UTC. Our SLA requires 99.9% uptime.",
            "urgent", "frustrated", "escalate_infrastructure", "medium"
        ),
        (
            "The email notification system seems to be down — none of our team has received any system emails in the past 3 hours including password resets and ticket notifications.",
            "high", "frustrated", "escalate_infrastructure", "medium"
        ),
        (
            "Your status page shows everything green but our customers are reporting they cannot submit support tickets through your embedded widget. Please investigate.",
            "urgent", "angry", "escalate_infrastructure", "hard"
        ),
        (
            "Data sync between your platform and our CRM integration has stopped. Last successful sync was 8 hours ago. We have 200 unsynced records accumulating.",
            "high", "frustrated", "escalate_engineering", "medium"
        ),
        (
            "The file upload feature is broken — all uploads fail with a generic error message. We need this working as we onboard a new client tomorrow morning.",
            "high", "frustrated", "escalate_infrastructure", "medium"
        ),
        (
            "We're seeing significant slowdown on the reporting module — queries that used to run in 2 seconds are now taking over 2 minutes. Noticed this after your maintenance window.",
            "medium", "frustrated", "escalate_engineering", "medium"
        ),
        (
            "The real-time collaboration feature in our shared workspace shows other users as offline even when they are actively working. This has made remote collaboration impossible.",
            "medium", "frustrated", "escalate_engineering", "hard"
        ),
    ],

    "enterprise_sales": [
        (
            "We are evaluating your platform for a 500-seat enterprise deployment. I'd like to schedule a technical deep-dive with your solutions engineering team this week.",
            "high", "positive", "assign_account_executive", "easy"
        ),
        (
            "Our legal team has reviewed your standard enterprise agreement and has 12 redline requests. Can you connect me with your contracts team to negotiate the terms?",
            "high", "neutral", "assign_account_executive", "medium"
        ),
        (
            "We're comparing your solution with two competitors for a major procurement decision due in 3 weeks. Can you provide a custom ROI analysis for a 200-person team?",
            "high", "neutral", "assign_account_executive", "medium"
        ),
        (
            "Our company processes data under HIPAA regulations. We need a signed BAA before we can move forward with the enterprise trial. Who is the right contact for this?",
            "high", "neutral", "assign_account_executive", "hard"
        ),
        (
            "We're interested in a custom on-premise deployment for our financial services firm. Our security team will not approve cloud-only solutions. Is this available?",
            "medium", "neutral", "assign_account_executive", "hard"
        ),
        (
            "I'd like to understand your enterprise SLA options. We need guaranteed 99.99% uptime with a 4-hour RTO for our mission-critical operations.",
            "medium", "neutral", "assign_account_executive", "medium"
        ),
        (
            "We're a nonprofit organization with 80 staff members. Do you offer discounted pricing for nonprofits? We have a very limited tech budget.",
            "low", "positive", "assign_account_executive", "easy"
        ),
        (
            "Our parent company already uses your platform with a 300-seat license. We'd like to add our subsidiary (150 seats) under the same contract. Who should I speak to?",
            "medium", "positive", "assign_account_executive", "medium"
        ),
    ],

    "bug_report": [
        (
            "Found a bug: when I type a special character (specifically the & symbol) in a ticket title, the entire form crashes and I lose all the content I've written.",
            "high", "frustrated", "create_bug_ticket", "medium"
        ),
        (
            "The ticket count in the sidebar shows 47 but when I open the list view there are only 31 tickets. The numbers are clearly out of sync after we archived some tickets.",
            "medium", "neutral", "create_bug_ticket", "medium"
        ),
        (
            "Reproducing consistently: if you click 'Save as Draft' and then immediately click 'Submit', the ticket is submitted twice, creating a duplicate entry in the system.",
            "high", "neutral", "create_bug_ticket", "hard"
        ),
        (
            "The time zone in our reports is wrong. My workspace is set to Tokyo (JST, UTC+9) but all timestamps in exports show UTC time without any conversion.",
            "medium", "frustrated", "create_bug_ticket", "medium"
        ),
        (
            "After the last update, the 'Assign to team member' dropdown no longer shows all team members — it cuts off at the first 10 alphabetically. We have 25 team members.",
            "medium", "frustrated", "create_bug_ticket", "easy"
        ),
        (
            "The notification badge on the Chrome browser tab doesn't clear after I've read all notifications. It keeps showing a count even when the notification panel is empty.",
            "low", "neutral", "create_bug_ticket", "easy"
        ),
        (
            "Security issue: I can view tickets from another workspace by modifying the ticket ID in the URL. This is a serious data isolation bug. Ticket IDs seem sequential.",
            "urgent", "neutral", "escalate_engineering", "hard"
        ),
        (
            "The 'Send Test Email' button in notification settings appears to work (shows success message) but no email is ever received. This makes it impossible to verify our email config.",
            "medium", "neutral", "create_bug_ticket", "medium"
        ),
    ],

    "cancellation": [
        (
            "I need to cancel my subscription at the end of this billing cycle. The product doesn't meet our current needs. Please confirm the cancellation and stop future charges.",
            "medium", "neutral", "retention_outreach", "easy"
        ),
        (
            "We are cancelling our account due to the recent price increase. The new pricing is 40% above what we started with and is no longer within our budget.",
            "medium", "frustrated", "retention_outreach", "medium"
        ),
        (
            "Please cancel my account immediately. I've found a competitor with better features at half the price. I should not be charged for next month.",
            "high", "angry", "retention_outreach", "easy"
        ),
        (
            "Our company is being acquired and we need to pause or cancel our subscription during the transition period. Can we put the account on hold for 90 days?",
            "medium", "neutral", "retention_outreach", "medium"
        ),
        (
            "We're a startup and just ran out of runway. We need to cancel all paid subscriptions immediately. Is there a free tier we could downgrade to temporarily?",
            "high", "frustrated", "retention_outreach", "medium"
        ),
        (
            "I signed up for a trial and forgot to cancel before it converted to a paid plan. I was charged $99. I'd like to cancel and receive a refund for the amount charged.",
            "medium", "frustrated", "refund_review", "easy"
        ),
        (
            "The decision has been made at the executive level to consolidate our tools and your platform was not selected. Please cancel the 25-seat enterprise account effective 30 days from today.",
            "high", "neutral", "retention_outreach", "medium"
        ),
        (
            "I'm moving to a competitor primarily because your mobile app is not good enough for my use case. If you ever improve the mobile experience I'll come back.",
            "low", "neutral", "retention_outreach", "easy"
        ),
    ],

    "integration_problem": [
        (
            "Our Salesforce integration is failing to sync contact records. The error log shows: 'Field mapping error: custom field CF_priority_level not found in target schema'. Salesforce version: Winter '24.",
            "high", "frustrated", "escalate_engineering", "hard"
        ),
        (
            "The Zapier integration stopped working after we updated our Zapier account. Old zaps still work but any new zap we try to create shows authentication failed.",
            "medium", "confused", "escalate_engineering", "medium"
        ),
        (
            "We're trying to set up the REST API integration for our custom CRM but the API documentation shows different endpoints than what's actually returning data. The docs appear outdated.",
            "medium", "frustrated", "escalate_engineering", "medium"
        ),
        (
            "The Jira integration is creating duplicate issues — every ticket submitted in your platform creates two Jira issues instead of one. Started happening after the Jira Cloud migration.",
            "high", "frustrated", "escalate_engineering", "hard"
        ),
        (
            "We need help setting up the OAuth 2.0 integration with our internal identity provider. Your docs only show examples for Google and Microsoft, not custom IdPs.",
            "medium", "neutral", "escalate_engineering", "hard"
        ),
        (
            "Data exported via your API is missing the 'custom_fields' array even though those fields are filled in and visible in the UI. This is breaking our downstream ETL pipeline.",
            "high", "frustrated", "escalate_engineering", "hard"
        ),
        (
            "Our Microsoft Teams bot integration doesn't post notifications to private channels, only public ones. This is a problem since our support team uses private channels.",
            "medium", "neutral", "escalate_engineering", "medium"
        ),
        (
            "The webhook payloads your system sends do not include the 'resolved_by' field even though it's listed in your documentation. We need this for our automated reporting.",
            "medium", "frustrated", "create_bug_ticket", "medium"
        ),
    ],
}

# ---------------------------------------------------------------------------
# Edge case tickets — 20 tickets designed to challenge AI models
# ---------------------------------------------------------------------------

EDGE_CASE_TICKETS = [
    # vague_message × 2
    {
        "message": "It's not working. Please help.",
        "category": "technical_issue",
        "priority": "medium",
        "sentiment": "frustrated",
        "expected_action": "escalate_engineering",
        "difficulty": "hard",
        "edge_case": "vague_message",
    },
    {
        "message": "Something is wrong with my account. I don't know what happened but it looks different now.",
        "category": "account_access",
        "priority": "medium",
        "sentiment": "confused",
        "expected_action": "manual_account_reset",
        "difficulty": "hard",
        "edge_case": "vague_message",
    },
    # angry_customer × 2
    {
        "message": "THIS IS ABSOLUTELY UNACCEPTABLE!!! I have been waiting 5 DAYS for a response and NOBODY is helping me! I pay $300/month for this garbage! I want to speak to a manager NOW!!!",
        "category": "billing",
        "priority": "urgent",
        "sentiment": "angry",
        "expected_action": "escalate_billing",
        "difficulty": "medium",
        "edge_case": "angry_customer",
    },
    {
        "message": "Your product is a scam. Plain and simple. I want every single cent refunded. Don't bother sending me another automated response. I'm contacting my bank for a chargeback.",
        "category": "refund",
        "priority": "urgent",
        "sentiment": "angry",
        "expected_action": "refund_review",
        "difficulty": "medium",
        "edge_case": "angry_customer",
    },
    # multiple_issues × 2
    {
        "message": "Hi, I have several problems I need help with: First, my invoice from last month is wrong (overcharged by $50). Second, I can't export my data to CSV — it errors out every time. Third, I've been trying to add a new team member for a week but the invite email never arrives.",
        "category": "billing",
        "priority": "high",
        "sentiment": "frustrated",
        "expected_action": "escalate_billing",
        "difficulty": "hard",
        "edge_case": "multiple_issues",
    },
    {
        "message": "We urgently need three things sorted: 1) Our Salesforce sync is broken since Tuesday, 2) Two of our users are locked out of their accounts, 3) We're being billed for 30 seats but only have 22 active users. Please prioritize all three.",
        "category": "integration_problem",
        "priority": "urgent",
        "sentiment": "angry",
        "expected_action": "escalate_engineering",
        "difficulty": "hard",
        "edge_case": "multiple_issues",
    },
    # missing_reference × 2
    {
        "message": "I need a refund for the charge from last month. I don't have the invoice number handy but it was definitely wrong.",
        "category": "refund",
        "priority": "medium",
        "sentiment": "neutral",
        "expected_action": "refund_review",
        "difficulty": "hard",
        "edge_case": "missing_reference",
    },
    {
        "message": "The bug I reported two weeks ago still isn't fixed. I thought someone was looking into it.",
        "category": "bug_report",
        "priority": "medium",
        "sentiment": "frustrated",
        "expected_action": "create_bug_ticket",
        "difficulty": "hard",
        "edge_case": "missing_reference",
    },
    # typo_heavy × 2
    {
        "message": "plese halp my acocunt is blokced and i cant login too the dashbord. i tred to reset my pasword but the emal didnt come. i hav importnat meetng tomorow",
        "category": "account_access",
        "priority": "high",
        "sentiment": "frustrated",
        "expected_action": "manual_account_reset",
        "difficulty": "medium",
        "edge_case": "typo_heavy",
    },
    {
        "message": "helo, my invoce form last mounth is rong. im being chargd $200 but it shoud be $100. pleaze fix this asap. thankss",
        "category": "billing",
        "priority": "medium",
        "sentiment": "frustrated",
        "expected_action": "escalate_billing",
        "difficulty": "medium",
        "edge_case": "typo_heavy",
    },
    # cancellation_threat × 2
    {
        "message": "If this issue isn't resolved by end of business today, I'm cancelling my subscription and moving to a competitor. I've been patient for too long.",
        "category": "technical_issue",
        "priority": "urgent",
        "sentiment": "angry",
        "expected_action": "escalate_engineering",
        "difficulty": "medium",
        "edge_case": "cancellation_threat",
    },
    {
        "message": "I'm seriously considering cancelling. The product was great 6 months ago but it's gotten slower and buggier with every update. Fix the performance issues or I'm out.",
        "category": "technical_issue",
        "priority": "high",
        "sentiment": "frustrated",
        "expected_action": "retention_outreach",
        "difficulty": "hard",
        "edge_case": "cancellation_threat",
    },
    # enterprise_escalation × 2
    {
        "message": "I am writing on behalf of our CTO, James Whitfield, regarding the ongoing performance degradation we have experienced over the past 72 hours. This has been escalated to executive level and we require a formal incident report, root cause analysis, and your VP of Engineering on a call by 5PM EST today. Our contract specifies a 4-hour SLA for P1 incidents.",
        "category": "outage",
        "priority": "urgent",
        "sentiment": "angry",
        "expected_action": "escalate_infrastructure",
        "difficulty": "hard",
        "edge_case": "enterprise_escalation",
    },
    {
        "message": "Per our enterprise agreement Section 4.2, we are formally notifying you of a Service Level breach. Uptime for the reporting module fell below 99.5% in March. We require a credit of 10% of monthly fees as specified in the SLA. Please have your legal team contact ours within 5 business days.",
        "category": "billing",
        "priority": "high",
        "sentiment": "neutral",
        "expected_action": "escalate_billing",
        "difficulty": "hard",
        "edge_case": "enterprise_escalation",
    },
    # unclear_ownership × 2
    {
        "message": "Hi, I was told to contact you about the thing with the dashboard. Someone from your team said they would fix it but I haven't heard back.",
        "category": "technical_issue",
        "priority": "medium",
        "sentiment": "confused",
        "expected_action": "escalate_engineering",
        "difficulty": "hard",
        "edge_case": "unclear_ownership",
    },
    {
        "message": "This is a follow-up to a previous conversation. I think I spoke with someone named Mike? Or maybe it was a different department. Anyway the original problem isn't resolved.",
        "category": "technical_issue",
        "priority": "medium",
        "sentiment": "confused",
        "expected_action": "escalate_engineering",
        "difficulty": "hard",
        "edge_case": "unclear_ownership",
    },
    # emotional_low_info × 2
    {
        "message": "I am so frustrated right now I don't even know where to start. This has been going on for days and I just can't deal with it anymore. Please someone help me.",
        "category": "technical_issue",
        "priority": "medium",
        "sentiment": "angry",
        "expected_action": "escalate_engineering",
        "difficulty": "hard",
        "edge_case": "emotional_low_info",
    },
    {
        "message": "I've spent 3 hours trying to figure this out and I'm completely exhausted. Your documentation doesn't help and the chat bot just loops me in circles.",
        "category": "technical_issue",
        "priority": "medium",
        "sentiment": "frustrated",
        "expected_action": "escalate_engineering",
        "difficulty": "medium",
        "edge_case": "emotional_low_info",
    },
    # urgent_ops × 2
    {
        "message": "PRODUCTION DOWN. API returning 503 across all endpoints. 1,200 active users affected. Revenue impact estimated at $8,000/hour. Need immediate escalation. On-call engineer contacted but needs your infrastructure access. This is a P0.",
        "category": "outage",
        "priority": "urgent",
        "sentiment": "angry",
        "expected_action": "escalate_infrastructure",
        "difficulty": "medium",
        "edge_case": "urgent_ops",
    },
    {
        "message": "Critical data loss event. Our automated backup job ran successfully at 02:00 UTC but when we tried to restore from it this morning, the backup file was corrupted/empty. We have a compliance audit in 4 hours and need access to the data immediately.",
        "category": "technical_issue",
        "priority": "urgent",
        "sentiment": "angry",
        "expected_action": "escalate_infrastructure",
        "difficulty": "hard",
        "edge_case": "urgent_ops",
    },
]

# ---------------------------------------------------------------------------
# Customer name and email pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Sarah", "James", "Emily", "Michael", "Olivia", "Daniel", "Sophia",
    "David", "Emma", "Chris", "Ava", "Ryan", "Isabella", "Matthew",
    "Mia", "Andrew", "Charlotte", "Joshua", "Amelia", "Kevin",
    "Elena", "Marcus", "Priya", "Ahmad", "Yuki", "Carlos", "Fatima",
    "Lucas", "Nina", "Thomas",
]

LAST_NAMES = [
    "Mitchell", "Chen", "Rodriguez", "Thompson", "Patel", "Williams",
    "Johnson", "Anderson", "Brown", "Taylor", "Wilson", "Moore",
    "Jackson", "Martin", "Garcia", "Lee", "Kowalski", "Müller",
    "Okonkwo", "Nakamura", "Fernandez", "Smith", "Davis", "Harris",
    "Clark", "Lewis", "Robinson", "Walker", "Hall", "Young",
]

EMAIL_DOMAINS = [
    "gmail.com", "company.io", "startup.co", "enterprise.com",
    "corp.net", "bizmail.com", "techfirm.io", "globalcorp.com",
    "mycompany.org", "work.io",
]


def generate_email(first: str, last: str, rng: random.Random) -> str:
    """Generate a realistic-looking email address."""
    domain = rng.choice(EMAIL_DOMAINS)
    style = rng.randint(0, 3)
    if style == 0:
        return f"{first.lower()}.{last.lower()}@{domain}"
    elif style == 1:
        return f"{first.lower()}{last.lower()[:3]}@{domain}"
    elif style == 2:
        return f"{first[0].lower()}{last.lower()}@{domain}"
    else:
        return f"{last.lower()}.{first.lower()}@{domain}"


# ---------------------------------------------------------------------------
# Build the full 100-ticket dataset
# ---------------------------------------------------------------------------

def build_dataset() -> list[dict]:
    rng = random.Random(42)  # local RNG — fully deterministic

    tickets = []
    idx = 1

    # --- 80 regular tickets ---
    for category, templates in REGULAR_TICKETS.items():
        for (message, priority, sentiment, expected_action, difficulty) in templates:
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            tickets.append({
                "ticket_id": f"SYN-{idx:04d}",
                "customer_name": f"{first} {last}",
                "email": generate_email(first, last, rng),
                "message": message,
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
                "expected_action": expected_action,
                "difficulty": difficulty,
                "language": "en",
                "edge_case": None,
            })
            idx += 1

    # --- 20 edge case tickets ---
    for ec in EDGE_CASE_TICKETS:
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        tickets.append({
            "ticket_id": f"SYN-{idx:04d}",
            "customer_name": f"{first} {last}",
            "email": generate_email(first, last, rng),
            "message": ec["message"],
            "category": ec["category"],
            "priority": ec["priority"],
            "sentiment": ec["sentiment"],
            "expected_action": ec["expected_action"],
            "difficulty": ec["difficulty"],
            "language": "en",
            "edge_case": ec["edge_case"],
        })
        idx += 1

    return tickets


# ---------------------------------------------------------------------------
# Build dataset_summary.json
# ---------------------------------------------------------------------------

def build_summary(tickets: list[dict]) -> dict:
    """Compute distribution statistics across all generated tickets."""
    categories = [t["category"] for t in tickets]
    priorities = [t["priority"] for t in tickets]
    sentiments = [t["sentiment"] for t in tickets]
    difficulties = [t["difficulty"] for t in tickets]
    edge_cases = [t["edge_case"] for t in tickets if t["edge_case"] is not None]
    edge_types = [t["edge_case"] for t in tickets if t["edge_case"] is not None]

    return {
        "total_tickets": len(tickets),
        "regular_tickets": len([t for t in tickets if t["edge_case"] is None]),
        "edge_case_tickets": len(edge_cases),
        "category_distribution": dict(Counter(categories)),
        "priority_distribution": dict(Counter(priorities)),
        "sentiment_distribution": dict(Counter(sentiments)),
        "difficulty_distribution": dict(Counter(difficulties)),
        "edge_case_distribution": dict(Counter(edge_types)),
        "language_distribution": dict(Counter(t["language"] for t in tickets)),
        "generated_with": "random.seed(42) — fully deterministic",
        "schema_version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Generating synthetic support ticket dataset...")
    tickets = build_dataset()

    # Save tickets
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(tickets)} tickets -> {TICKETS_FILE}")

    # Save summary
    summary = build_summary(tickets)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved dataset summary -> {SUMMARY_FILE}")

    # Print quick stats
    print("\n--- Dataset Summary ---")
    print(f"   Total tickets : {summary['total_tickets']}")
    print(f"   Regular       : {summary['regular_tickets']}")
    print(f"   Edge cases    : {summary['edge_case_tickets']}")
    print(f"\n   Category distribution:")
    for k, v in sorted(summary["category_distribution"].items()):
        print(f"     {k:<25} {v:>3}")
    print(f"\n   Priority distribution:")
    for k, v in sorted(summary["priority_distribution"].items()):
        print(f"     {k:<15} {v:>3}")
    print(f"\n   Difficulty distribution:")
    for k, v in sorted(summary["difficulty_distribution"].items()):
        print(f"     {k:<10} {v:>3}")

    print("\n[DONE] Dataset ready for AI evaluation pipeline.")


if __name__ == "__main__":
    main()
