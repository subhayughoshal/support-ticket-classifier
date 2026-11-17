"""Bonus task - rule-based suggested replies (no external API, no keys needed)."""

RESPONSES = {
    "Login Issue": (
        "Please use the 'Forgot Password' option on the login page to reset your password. "
        "If your account is locked, wait 15 minutes and try again. If the problem continues, "
        "please contact the support team."),
    "Application Error": (
        "We are sorry for the trouble. Please note the exact error message (a screenshot helps), "
        "refresh the page and try again. Our technical team has been notified and will investigate."),
    "Report": (
        "You can generate or download reports from the Reports section. If you need a custom "
        "report or notice wrong figures, please share the date range and filters and we will help."),
    "Account Update": (
        "You can update your profile details from Account Settings > Profile. For security reasons, "
        "changes to mobile number or email may need OTP verification. Contact support if you need help."),
    "Performance": (
        "Sorry for the slowness. Please try clearing your browser cache or using another browser. "
        "We are checking our servers for performance issues and will update you shortly."),
}


def suggested_response(category: str) -> str:
    return RESPONSES.get(category, "Thank you for contacting support. We will get back to you shortly.")
