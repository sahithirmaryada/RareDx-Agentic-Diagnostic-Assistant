import logging
import random
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Statistical patterns for lab markers [cite: 27, 87]
DISEASE_PROFILES = {
    "Pompe Disease": {
        "marker": "Creatine Kinase",
        "unit": "U/L",
        "range": (500, 2000),  # Typically elevated
        "loinc": "2157-6"
    }
}

def generate_fhir_record(disease_name):
    """Returns a FHIR-formatted JSON record [cite: 31, 91]"""
    profile = DISEASE_PROFILES.get(disease_name)
    if not profile:
        return {"error": "Disease profile not found"}

    # Sample a random value from the expected distribution [cite: 30, 90]
    simulated_value = round(random.uniform(profile["range"][0], profile["range"][1]), 2)

    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": profile["loinc"]}]
        },
        "effectiveDateTime": datetime.now().isoformat(),
        "valueQuantity": {
            "value": simulated_value,
            "unit": profile["unit"]
        }
    }

if __name__ == "__main__":
    logger.info(json.dumps(generate_fhir_record("Pompe Disease"), indent=2))