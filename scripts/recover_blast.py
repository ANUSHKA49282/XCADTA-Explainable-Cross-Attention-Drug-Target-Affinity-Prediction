
import urllib.parse
import urllib.request
import time
from pathlib import Path

# Project folder
ROOT = Path(__file__).resolve().parent.parent

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# Enter the previous BLAST request ID
RID = input("Enter your previous BLAST RID: ").strip()

if not RID:
    raise SystemExit("RID cannot be empty.")

params = {
    "CMD": "Get",
    "RID": RID,
    "FORMAT_TYPE": "XML"
}

url = BLAST_URL + "?" + urllib.parse.urlencode(params)

output_file = ROOT / "results/case_studies/protein_blast_results.xml"
output_file.parent.mkdir(parents=True, exist_ok=True)

print("\nChecking existing BLAST job...")

for attempt in range(8):

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "XCADTA-Protein-Identity-Audit/1.0"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            result = response.read().decode("utf-8", errors="replace")

        if "Status=WAITING" in result:
            print("BLAST still running:", attempt + 1, "/8")
            time.sleep(30)
            continue

        if "Status=FAILED" in result or "Status=UNKNOWN" in result:
            raise SystemExit("Previous BLAST job failed or expired.")

        if "<BlastOutput" in result:
            output_file.write_text(result, encoding="utf-8")

            print("\nBLAST results retrieved successfully!")
            print("Saved:", output_file)
            break

        print("NCBI response:", result[:500])
        time.sleep(30)

    except Exception as error:
        print("Request error:", error)
        break

else:
    print("BLAST is still processing. Try again later.")
