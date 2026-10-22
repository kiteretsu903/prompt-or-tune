"""Ground-truth business policy used to label synthetic tickets."""

POLICY_TEXT = """
Classify each support ticket using exactly one rule:

1. Credential theft, suspicious login, or stolen password:
   category=security_incident, priority=P0, escalate=true,
   reason_code=credential_theft.
2. Data deletion, data corruption, or repeated loss of saved work:
   category=bug_report, priority=P0, escalate=true, reason_code=data_loss.
3. A duplicate charge where each charge is over $500:
   category=billing_refund, priority=P1, escalate=true,
   reason_code=duplicate_charge_over_500.
4. A normal refund, cancellation, or billing correction that is not rule 3:
   category=billing_refund, priority=P2, escalate=false,
   reason_code=ordinary_refund.
5. A software defect that blocks a core workflow but does not lose data:
   category=bug_report, priority=P1, escalate=true,
   reason_code=service_blocking_bug.
6. A visual or cosmetic defect that does not block work:
   category=bug_report, priority=P3, escalate=false, reason_code=cosmetic_bug.
7. A request for a new capability or product enhancement:
   category=feature_request, priority=P3, escalate=false,
   reason_code=product_enhancement.
8. A legitimate user who cannot access a locked account:
   category=account_access, priority=P2, escalate=false,
   reason_code=locked_account.
9. A general how-to or product-information question:
   category=general_question, priority=P3, escalate=false,
   reason_code=general_information.
""".strip()
