"""Which claim fields the models use, how they are encoded, and plain-language reasons. Author: Vivek Mitra"""
import pandas as pd

TARGET = "FraudFound_P"

# Ordered bands become ranks for gradient boosting (order from the band's meaning).
ORDINAL = {
    "VehiclePrice": ["less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000", "60000 to 69000", "more than 69000"],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": ["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"],
    "AgeOfPolicyHolder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
}
NUMERIC = ["Age", "Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed"]
LABELS = ["Make", "AccidentArea", "Fault", "PolicyType", "VehicleCategory", "BasePolicy", "PoliceReportFiled",
          "WitnessPresent", "AgentType", "RepNumber", "Month", "MonthClaimed", "DayOfWeek", "DayOfWeekClaimed",
          "Sex", "MaritalStatus"]
CALENDAR = ["Month", "MonthClaimed", "DayOfWeek", "DayOfWeekClaimed", "WeekOfMonth", "WeekOfMonthClaimed"]
SENSITIVE = ["Sex", "MaritalStatus"]   # never in the main models; one check adds them back
# Candidate field sets, chosen on 1995 (never on 1996).
SETS = {"all_fields": [], "no_calendar": CALENDAR, "no_calendar_no_rep": CALENDAR + ["RepNumber"]}


def columns(set_name, sensitive=False):
    drop = set(SETS[set_name]) | (set() if sensitive else set(SENSITIVE))
    return [c for c in list(ORDINAL) + NUMERIC + LABELS if c not in drop]


def lr_x(d, cols):
    """Logistic regression sees every field as text; a one-hot encoder turns each value into a 0/1 column."""
    return d[cols].astype(str)


def gbm_x(d, cols, ref):
    """Gradient boosting: bands as ranks, numbers as numbers, labels as categories taken from the training data `ref`."""
    X = pd.DataFrame(index=d.index)
    for c in cols:
        if c in ORDINAL:
            X[c] = d[c].map({v: i for i, v in enumerate(ORDINAL[c])})
        elif c in NUMERIC:
            X[c] = d[c]
        else:
            X[c] = pd.Categorical(d[c].astype(str), categories=sorted(ref[c].astype(str).unique()))
    return X


PLAIN = {
    "Fault": lambda v: "Policy holder was at fault" if v == "Policy Holder" else "Third party was at fault",
    "BasePolicy": lambda v: f"Cover type: {v.lower()}",
    "PolicyType": lambda v: f"Policy: {v.lower()}",
    "VehicleCategory": lambda v: f"Vehicle type: {v.lower()}",
    "VehiclePrice": lambda v: "Vehicle price " + v.replace("less than", "under").replace("more than", "over").replace(" to ", "–") + " dollars",
    "Make": lambda v: f"Car make: {v}",
    "AccidentArea": lambda v: f"{v} accident",
    "PastNumberOfClaims": lambda v: "No past claims" if v == "none" else f"Past claims: {v}",
    "AddressChange_Claim": lambda v: "No recent address change" if v == "no change" else f"Address changed {v} before the claim",
    "AgeOfVehicle": lambda v: "New vehicle" if v == "new" else f"Vehicle age: {v}",
    "AgeOfPolicyHolder": lambda v: f"Policy holder aged {v}",
    "Age": lambda v: "Driver age not recorded" if int(v) == 0 else f"Driver age {int(v)}",
    "NumberOfSuppliments": lambda v: "No supplements on the claim" if v == "none" else f"Supplements on the claim: {v}",
    "NumberOfCars": lambda v: f"Cars on the policy: {v}",
    "Days_Policy_Accident": lambda v: "Accident on the day the policy started" if v == "none" else f"Accident {v} days after the policy started",
    "Days_Policy_Claim": lambda v: "Claim on the day the policy started" if v == "none" else f"Claim {v} days after the policy started",
    "PoliceReportFiled": lambda v: "No police report filed" if v == "No" else "Police report filed",
    "WitnessPresent": lambda v: "No witness present" if v == "No" else "Witness present",
    "AgentType": lambda v: f"Sold through an {v.lower()} agent",
    "RepNumber": lambda v: f"Handled by claims rep no. {v}",
    "Deductible": lambda v: f"Deductible ${int(v)}",
    "DriverRating": lambda v: f"Driver rating {int(v)} of 4",
}


def reason(col, value):
    return PLAIN[col](value) if col in PLAIN else f"{col}: {value}"
