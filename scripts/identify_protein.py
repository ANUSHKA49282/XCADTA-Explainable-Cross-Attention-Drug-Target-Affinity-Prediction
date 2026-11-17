
import csv
import time
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Project folder
ROOT = Path(__file__).resolve().parent.parent

FASTA_FILE = ROOT / "results/case_studies/candidate_protein_new.fasta"
OUTPUT_FILE = ROOT / "results/case_studies/protein_identity_results_new.csv"

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# Check FASTA file
if not FASTA_FILE.exists():
    raise SystemExit("FASTA file not found: " + str(FASTA_FILE))

sequence_text = FASTA_FILE.read_text().strip()

sequence = ""

for line in sequence_text.splitlines():
    if not line.startswith(">"):
        sequence += line.strip()

if not sequence:
    raise SystemExit("FASTA sequence is empty.")

print("XCADTA PROTEIN IDENTIFICATION")
print("Sequence length:", len(sequence))
print("Submitting sequence to NCBI BLAST...")

# Send request to NCBI
def request_blast(params):
    data = urllib.parse.urlencode(params).encode("utf-8")

    request = urllib.request.Request(
        BLAST_URL,
        data=data,
        headers={"User-Agent": "XCADTA-Protein-Identity-Audit/1.0"}
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


# Submit BLASTP job
submission = request_blast({
    "CMD": "Put",
    "PROGRAM": "blastp",
    "DATABASE": "nr",
    "QUERY": sequence_text,
    "HITLIST_SIZE": "10",
    "EXPECT": "10",
    "FORMAT_TYPE": "XML"
})

rid_match = re.search(r"RID\s*=\s*(\S+)", submission)
rtoe_match = re.search(r"RTOE\s*=\s*(\d+)", submission)

if not rid_match:
    print(submission[:2000])
    raise SystemExit("Could not obtain BLAST request ID.")

rid = rid_match.group(1)
wait_time = int(rtoe_match.group(1)) if rtoe_match else 20

print("BLAST request ID:", rid)
print("Initial wait:", wait_time, "seconds")

time.sleep(wait_time)

# Check BLAST results
for attempt in range(40):

    response = request_blast({
        "CMD": "Get",
        "RID": rid,
        "FORMAT_TYPE": "XML"
    })

    if "Status=WAITING" in response:
        print("BLAST running...", attempt + 1)
        time.sleep(15)
        continue

    if "Status=FAILED" in response or "Status=UNKNOWN" in response:
        raise SystemExit("BLAST job failed or expired.")

    if "Status=READY" in response or "<BlastOutput" in response:
        break

    print("Waiting for BLAST results...", attempt + 1)
    time.sleep(15)

else:
    raise SystemExit("BLAST timed out. Please try again later.")

# Read BLAST XML results
try:
    root = ET.fromstring(response)
except ET.ParseError:
    print(response[:2000])
    raise SystemExit("Could not read BLAST results.")

results = []

for hit in root.findall(".//Hit"):

    hit_id = hit.findtext("Hit_id", "")
    accession = hit.findtext("Hit_accession", "")
    description = hit.findtext("Hit_def", "")

    hsps = hit.findall("./Hit_hsps/Hsp")

    if not hsps:
        continue

    # Select the highest-scoring alignment
    hsp = max(
        hsps,
        key=lambda item: float(item.findtext("Hsp_bit-score", "0"))
    )

    identity = int(hsp.findtext("Hsp_identity", "0"))
    alignment_length = int(hsp.findtext("Hsp_align-len", "1"))
    query_from = int(hsp.findtext("Hsp_query-from", "1"))
    query_to = int(hsp.findtext("Hsp_query-to", "1"))
    evalue = float(hsp.findtext("Hsp_evalue", "999"))

    percent_identity = 100 * identity / alignment_length
    query_coverage = 100 * (query_to - query_from + 1) / len(sequence)

    results.append({
        "accession": accession,
        "description": description,
        "identity_percent": round(percent_identity, 2),
        "query_coverage_percent": round(query_coverage, 2),
        "evalue": evalue,
        "hit_id": hit_id
    })

if not results:
    raise SystemExit("No protein matches were returned.")

# Save results
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "accession",
            "description",
            "identity_percent",
            "query_coverage_percent",
            "evalue",
            "hit_id"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

# Display top matches
print("\n--TOP PROTEIN MATCHES--")

for i, result in enumerate(results, start=1):

    print("\nMatch", i)
    print("Protein:", result["description"])
    print("Accession:", result["accession"])
    print("Identity:", result["identity_percent"], "%")
    print("Query coverage:", result["query_coverage_percent"], "%")
    print("E-value:", result["evalue"])

print("\nResults saved:", OUTPUT_FILE)
