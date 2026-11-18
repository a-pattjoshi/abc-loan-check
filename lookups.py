"""Lookup tables the chatbot needs: pincode -> tier and the vehicle menu."""
import json

import pandas as pd

import config as C


def build():
    d = pd.read_parquet(C.CLEAN_PARQUET)
    pin = d.dropna(subset=["Pincode", "Final_Tier"])
    pin_tier = pin.groupby("Pincode")["Final_Tier"].first()
    pin3 = pin.assign(p3=pin["Pincode"].astype(str).str[:3])
    pin3_tier = pin3.groupby("p3")["Final_Tier"].agg(lambda s: s.value_counts().index[0])
    veh = (d.dropna(subset=["Model_Description", "Make_Code", "Product_Code", "Model_Variant"])
           .groupby("Model_Description")[["Make_Code", "Product_Code", "Model_Variant"]].first().reset_index())
    out = {
        "pin_tier": {str(int(k)): v for k, v in pin_tier.items()},
        "pin3_tier": pin3_tier.to_dict(),
        "vehicles": [{"desc": r.Model_Description, "make": r.Make_Code, "product": r.Product_Code,
                      "variant": r.Model_Variant} for r in veh.itertuples()],
    }
    C.LOOKUPS_PATH.write_text(json.dumps(out))
    return out


def load():
    if not C.LOOKUPS_PATH.exists():
        return build()
    return json.loads(C.LOOKUPS_PATH.read_text())


if __name__ == "__main__":
    o = build()
    print(len(o["pin_tier"]), "pincodes,", len(o["vehicles"]), "vehicles")
