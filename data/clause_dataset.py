"""
clause_dataset.py – Curated benchmark dataset of labelled legal contract clauses.

Used for:
  • Training the scikit-learn TF-IDF + Logistic Regression baseline classifier
  • Evaluating classification metrics (Precision, Recall, F1, Confusion Matrix)
  • Offline testing and CI validation without network dependencies
"""

from typing import Dict, List, Tuple

# Core categories supported across the system
BENCHMARK_CATEGORIES = [
    "Payment",
    "Termination",
    "Liability",
    "Indemnity",
    "Confidentiality",
    "Obligations",
    "Renewal",
    "Dispute Resolution",
    "Non-Compete",
    "Intellectual Property",
    "Other / Unknown",
]

# Curated dataset of typical contract clauses across commercial & employment agreements
LABELED_CLAUSES: List[Tuple[str, str]] = [
    # ── Payment ──────────────────────────────────────────────────────────
    (
        "Client shall pay all invoiced amounts within thirty (30) days of the date of invoice. Late payments shall accrue interest at 1.5% per month.",
        "Payment",
    ),
    (
        "The fees for the Professional Services shall be billed on a monthly time and materials basis in accordance with Exhibit A rate card.",
        "Payment",
    ),
    (
        "Company will reimburse Contractor for all reasonable and documented out-of-pocket travel and lodging expenses incurred in performing the Services.",
        "Payment",
    ),
    (
        "Customer agrees to pay a non-refundable upfront deposit of $25,000 upon execution of this Statement of Work.",
        "Payment",
    ),
    (
        "All fees quoted herein are exclusive of applicable value-added, sales, use, or withholding taxes, which shall be added to the invoice.",
        "Payment",
    ),

    # ── Termination ──────────────────────────────────────────────────────
    (
        "Either party may terminate this Agreement without cause upon giving ninety (90) days prior written notice to the other party.",
        "Termination",
    ),
    (
        "Either party may terminate this Agreement immediately upon written notice if the other party commits a material breach and fails to cure within 30 days.",
        "Termination",
    ),
    (
        "Upon termination or expiration of this Agreement, Licensee shall immediately cease all use of the Licensed Software and return all Confidential Information.",
        "Termination",
    ),
    (
        "Company may terminate Contractor's engagement immediately for cause in the event of gross negligence, fraud, or conviction of a felony.",
        "Termination",
    ),
    (
        "In the event of insolvency, bankruptcy, or assignment for the benefit of creditors, the non-defaulting party may terminate immediately.",
        "Termination",
    ),

    # ── Liability ────────────────────────────────────────────────────────
    (
        "In no event shall either party's aggregate cumulative liability arising out of or related to this Agreement exceed the total fees paid in the prior twelve months.",
        "Liability",
    ),
    (
        "Neither party shall be liable to the other for any indirect, incidental, special, punitive, exemplary, or consequential damages of any kind.",
        "Liability",
    ),
    (
        "Provider's maximum liability for any data loss, system outage, or service interruption shall be strictly capped at the monthly subscription fee.",
        "Liability",
    ),
    (
        "Nothing in this Section shall limit or exclude either party's liability for death, personal injury caused by negligence, or fraudulent misrepresentation.",
        "Liability",
    ),
    (
        "The limitations on liability set forth herein shall apply regardless of whether the claim sounds in contract, tort, warranty, or strict liability.",
        "Liability",
    ),

    # ── Indemnity ────────────────────────────────────────────────────────
    (
        "Vendor shall defend, indemnify, and hold harmless Customer from and against any third-party claims alleging that the Software infringes any patent or copyright.",
        "Indemnity",
    ),
    (
        "Customer shall indemnify and hold harmless Provider against all claims, liabilities, and expenses arising out of Customer's breach of applicable privacy laws.",
        "Indemnity",
    ),
    (
        "The indemnifying party's obligations are conditioned upon the indemnified party giving prompt written notice of any claim and sole control of defense.",
        "Indemnity",
    ),
    (
        "Each party agrees to indemnify the other from damages resulting from the gross negligence or willful misconduct of the indemnifying party's personnel.",
        "Indemnity",
    ),
    (
        "Service Provider shall hold harmless and defend the Client, its officers, directors, and employees against any third-party lawsuits.",
        "Indemnity",
    ),

    # ── Confidentiality ──────────────────────────────────────────────────
    (
        "Recipient agrees to hold all Confidential Information in strict confidence and use the same degree of care it uses for its own confidential data, but not less than reasonable care.",
        "Confidentiality",
    ),
    (
        "Confidential Information shall not include information that is publicly known, already known to Recipient without restriction, or independently developed.",
        "Confidentiality",
    ),
    (
        "The obligations of confidentiality and non-disclosure under this Section shall survive termination of this Agreement for five (5) years.",
        "Confidentiality",
    ),
    (
        "Neither party shall disclose the existence or terms of this Agreement to any third party without the prior written consent of the other party.",
        "Confidentiality",
    ),
    (
        "Recipient may disclose Confidential Information solely to employees, contractors, and legal advisors who have a strict need to know and are bound by confidentiality.",
        "Confidentiality",
    ),

    # ── Obligations ──────────────────────────────────────────────────────
    (
        "Vendor represents and warrants that the Deliverables shall conform in all material respects with the technical specifications set forth in Schedule 1.",
        "Obligations",
    ),
    (
        "Contractor covenants that it possesses the requisite professional skill, experience, and certifications necessary to perform the Services.",
        "Obligations",
    ),
    (
        "Customer agrees to provide timely access to its facilities, technical personnel, and data necessary for Consultant to complete the Statement of Work.",
        "Obligations",
    ),
    (
        "Supplier shall maintain commercial general liability insurance and workers' compensation coverage in amounts of not less than $2,000,000 per occurrence.",
        "Obligations",
    ),
    (
        "Distributor shall use commercially reasonable efforts to promote, market, and distribute the Products throughout the Territory.",
        "Obligations",
    ),

    # ── Renewal ──────────────────────────────────────────────────────────
    (
        "This Agreement shall automatically renew for additional successive terms of one (1) year each, unless either party gives written notice of non-renewal at least 60 days prior.",
        "Renewal",
    ),
    (
        "The Initial Term shall automatically extend for renewal periods equal in length to the initial term unless opted out in writing prior to expiration.",
        "Renewal",
    ),
    (
        "Unless Customer notifies Provider in writing at least thirty (30) days prior to the end of the current term, the subscription will auto-renew at standard list rates.",
        "Renewal",
    ),
    (
        "This Lease shall automatically renew on a month-to-month basis upon expiration of the primary term until terminated by either party.",
        "Renewal",
    ),

    # ── Dispute Resolution ───────────────────────────────────────────────
    (
        "This Agreement and all claims arising out of or related to it shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflicts of law.",
        "Dispute Resolution",
    ),
    (
        "Any dispute, controversy, or claim arising under this contract shall be submitted to binding arbitration administered by the American Arbitration Association in New York, NY.",
        "Dispute Resolution",
    ),
    (
        "The parties irrevocably submit to the exclusive jurisdiction of the federal and state courts situated in London, England for any action or proceeding.",
        "Dispute Resolution",
    ),
    (
        "The parties agree to attempt in good faith to resolve any dispute through informal executive escalation and mediation prior to commencing legal proceedings.",
        "Dispute Resolution",
    ),
    (
        "Each party hereby waives any right to a trial by jury in any lawsuit, proceeding, or counterclaim arising out of this Agreement.",
        "Dispute Resolution",
    ),

    # ── Non-Compete ──────────────────────────────────────────────────────
    (
        "During the Term and for a period of eighteen (18) months thereafter, Employee shall not engage in, manage, or advise any competitive business within the United States.",
        "Non-Compete",
    ),
    (
        "Consultant agrees not to solicit, recruit, or hire any employee or contractor of Client for a period of twelve (12) months following contract completion.",
        "Non-Compete",
    ),
    (
        "Contractor covenants not to solicit, divert, or do business with any client or customer of Company with whom Contractor had direct contact during the engagement.",
        "Non-Compete",
    ),
    (
        "The restrictive covenants set forth in this Section are acknowledged to be reasonable and necessary to protect the legitimate goodwill and trade secrets of the Company.",
        "Non-Compete",
    ),

    # ── Intellectual Property ────────────────────────────────────────────
    (
        "Contractor hereby assigns to Client all right, title, and interest worldwide in and to all inventions, code, designs, and Deliverables created under this Agreement.",
        "Intellectual Property",
    ),
    (
        "All work product created by Consultant shall be deemed a 'work made for hire' for the benefit of Company under U.S. copyright law.",
        "Intellectual Property",
    ),
    (
        "Provider grants Customer a non-exclusive, non-transferable, revocable license to access and use the Cloud Software solely for Customer's internal operations.",
        "Intellectual Property",
    ),
    (
        "Each party retains all right, title, and ownership in its pre-existing intellectual property, background technology, and proprietary tools.",
        "Intellectual Property",
    ),
    (
        "Customer shall not reverse engineer, decompile, disassemble, or derive the source code of the Provider's proprietary software.",
        "Intellectual Property",
    ),

    # ── Other / Unknown ──────────────────────────────────────────────────
    (
        "This Agreement constitutes the entire agreement between the parties and supersedes all prior oral or written negotiations, agreements, and representations.",
        "Other / Unknown",
    ),
    (
        "All notices required under this Agreement shall be in writing and delivered by certified mail or nationally recognized overnight courier to the address listed above.",
        "Other / Unknown",
    ),
    (
        "If any provision of this Agreement is held to be invalid or unenforceable, such provision shall be severed and the remaining provisions shall continue in full force.",
        "Other / Unknown",
    ),
    (
        "Neither party may assign or transfer this Agreement without the prior written consent of the other party, except in connection with a merger or acquisition.",
        "Other / Unknown",
    ),
    (
        "IN WITNESS WHEREOF, the parties hereto have caused this Agreement to be executed by their duly authorized representatives as of the Effective Date.",
        "Other / Unknown",
    ),
]


def get_benchmark_dataset() -> Tuple[List[str], List[str]]:
    """Return texts and corresponding category labels for ML training/evaluation."""
    texts = [item[0] for item in LABELED_CLAUSES]
    labels = [item[1] for item in LABELED_CLAUSES]
    return texts, labels
