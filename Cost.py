from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Constants (molar masses, g/mol)
# -----------------------------
M = {
    "Li": 6.94,
    "O": 16.00,
    "Ni": 58.693,
    "Mn": 54.938,
    "Co": 58.933,
    "H2O": 18.01528,
    "Li2CO3": 73.891,
    "LiOH_H2O": 41.958,
    "Li2O": 29.88,   # 2*Li + O
    "NaOH": 40.000,
    "Na2SO4": 142.04,
    "CO2": 44.01,
}

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class Chemistry:
    x_Ni: float = 0.8
    y_Mn: float = 0.1
    z_Co: float = 0.1

@dataclass
class LiRoute:
    route: str = "hydroxide"  # "hydroxide" or "oxide" or "carbonate"
    f_li: float = 1.02
    u_li: float = 1.00

@dataclass
class CathodeFormulation:
    w_cam: float = 0.96
    w_cb: float = 0.02
    w_pvdf: float = 0.02
    solids_loading: float = 0.60
    nmp_recovery: float = 0.95

@dataclass
class Prices:
    li2co3: float = 96.0
    lioh_h2o: float = 11.10
    li2o: float = 29.15
    naoh: float = 0.8
    ni_metal: float = 11.24
    mn_metal: float = 1.89
    co_metal: float = 38.24
    cb: float = 4.0
    pvdf: float = 20.0
    nmp: float = 3.0
    elec_per_kwh: float = 0.07
    gas_per_therm: float = 0.90
    water_per_gal: float = 0.0030

@dataclass
class UtilitiesConsumption:
    electricity_kwh_per_kg: float = 1.5
    gas_therm_per_kg: float = 0.02
    water_gal_per_kg: float = 0.1

@dataclass
class EquipmentScaling:
    baseline_musd: Dict[str, float] = None
    exponents: Dict[str, float] = None
    S_ref_GWh: float = 50.0
    S_target_GWh: float = 50.0
    life_years: float = 10.0
    annual_output_kg: Optional[float] = None

# -----------------------------
# Core chemistry
# -----------------------------
def cam_molar_mass(chem: Chemistry) -> float:
    """Li(Ni_x Mn_y Co_z)O2"""
    mM = chem.x_Ni * M["Ni"] + chem.y_Mn * M["Mn"] + chem.z_Co * M["Co"]
    return M["Li"] + mM + 2 * M["O"]

def n_cam_per_kg(chem: Chemistry) -> float:
    return 1000.0 / cam_molar_mass(chem)

def reagent_masses_per_kg_cam(chem: Chemistry, li: LiRoute) -> Dict[str, float]:
    n_cam = n_cam_per_kg(chem)
    out: Dict[str, float] = {}

    out["Ni_metal_g"] = chem.x_Ni * M["Ni"] * n_cam
    out["Mn_metal_g"] = chem.y_Mn * M["Mn"] * n_cam
    out["Co_metal_g"] = chem.z_Co * M["Co"] * n_cam

    out["NaOH_g"] = 2.0 * n_cam * M["NaOH"]
    out["Na2SO4_g"] = 1.0 * n_cam * M["Na2SO4"]

    route = li.route.lower()
    if route == "carbonate":
        n_li2co3 = 0.5 * n_cam * li.f_li * li.u_li
        out["Li2CO3_g"] = n_li2co3 * M["Li2CO3"]
        out["LiOH_H2O_g"] = 0.0
        out["Li2O_g"] = 0.0
        out["CO2_g"] = 0.5 * n_cam * M["CO2"]
        out["H2O_from_reaction_g"] = 1.0 * n_cam * M["H2O"]
    elif route == "hydroxide":
        n_lioh = 1.0 * n_cam * li.f_li * li.u_li
        out["Li2CO3_g"] = 0.0
        out["LiOH_H2O_g"] = n_lioh * M["LiOH_H2O"]
        out["Li2O_g"] = 0.0
        out["CO2_g"] = 0.0
        out["H2O_from_reaction_g"] = 2.5 * n_cam * M["H2O"]
    else:  # oxide
        n_li2o = 0.5 * n_cam * li.f_li * li.u_li
        out["Li2CO3_g"] = 0.0
        out["LiOH_H2O_g"] = 0.0
        out["Li2O_g"] = n_li2o * M["Li2O"]
        out["CO2_g"] = 0.0
        out["H2O_from_reaction_g"] = 1.0 * n_cam * M["H2O"]

    return out

def cathode_extras_per_kg_cam(cf: CathodeFormulation) -> Dict[str, float]:
    if cf.w_cam <= 0 or cf.w_cam > 1:
        raise ValueError("w_cam must be in (0,1]")

    m_cb = 1000.0 * (cf.w_cb / cf.w_cam)
    m_pvdf = 1000.0 * (cf.w_pvdf / cf.w_cam)

    m_solids = 1000.0 + m_cb + m_pvdf
    if not (0 < cf.solids_loading < 1):
        raise ValueError("solids_loading must be in (0,1)")

    m_nmp = m_solids * (1.0 - cf.solids_loading) / cf.solids_loading
    m_nmp_makeup = m_nmp * (1.0 - cf.nmp_recovery)

    return {"CB_g": m_cb, "PVDF_g": m_pvdf, "NMP_makeup_g": m_nmp_makeup}

# -----------------------------
# Cost layers
# -----------------------------
def materials_cost_per_kg(cam_reagents_g: Dict[str, float],
                          cathode_extras_g: Dict[str, float],
                          prices: Prices,
                          li: LiRoute) -> Dict[str, float]:
    g2kg = 1.0 / 1000.0
    route = li.route.lower()

    if route == "carbonate":
        c_li = cam_reagents_g["Li2CO3_g"] * g2kg * prices.li2co3
    elif route == "hydroxide":
        c_li = cam_reagents_g["LiOH_H2O_g"] * g2kg * prices.lioh_h2o
    else:
        c_li = cam_reagents_g["Li2O_g"] * g2kg * prices.li2o

    c_naoh = cam_reagents_g["NaOH_g"] * g2kg * prices.naoh
    c_ni = cam_reagents_g["Ni_metal_g"] * g2kg * prices.ni_metal
    c_mn = cam_reagents_g["Mn_metal_g"] * g2kg * prices.mn_metal
    c_co = cam_reagents_g["Co_metal_g"] * g2kg * prices.co_metal

    c_cb = cathode_extras_g["CB_g"] * g2kg * prices.cb
    c_pvdf = cathode_extras_g["PVDF_g"] * g2kg * prices.pvdf
    c_nmp = cathode_extras_g["NMP_makeup_g"] * g2kg * prices.nmp

    total = c_li + c_naoh + c_ni + c_mn + c_co + c_cb + c_pvdf + c_nmp
    return {
        "Li_source": c_li,
        "NaOH": c_naoh,
        "Ni": c_ni,
        "Mn": c_mn,
        "Co": c_co,
        "CB": c_cb,
        "PVDF": c_pvdf,
        "NMP_makeup": c_nmp,
        "materials_total": total,
    }

def utilities_cost_per_kg(u: UtilitiesConsumption, p: Prices) -> Dict[str, float]:
    c_e = u.electricity_kwh_per_kg * p.elec_per_kwh
    c_g = u.gas_therm_per_kg * p.gas_per_therm
    c_w = u.water_gal_per_kg * p.water_per_gal
    return {
        "electricity": c_e,
        "gas": c_g,
        "water": c_w,
        "utilities_total": c_e + c_g + c_w,
    }

def equipment_cost_per_kg(eq: EquipmentScaling) -> Dict[str, float]:
    """
    total_capex_musd: M$ at target scale
    annualized_capex_musd: M$/yr
    equipment_total: $/kg
    """
    if not eq.baseline_musd:
        return {
            "equipment_total": 0.0,
            "capex_total_musd": 0.0,
            "annualized_capex_musd": 0.0,
        }

    S = max(eq.S_target_GWh, 1e-9)
    Sref = max(eq.S_ref_GWh, 1e-9)

    total_capex_musd = 0.0
    for k, cref in eq.baseline_musd.items():
        b = (eq.exponents or {}).get(k, 0.6)
        total_capex_musd += cref * (S / Sref) ** b

    life = max(eq.life_years, 1e-9)
    annualized_capex_musd = total_capex_musd / life
    annualized_capex_usd = annualized_capex_musd

    if not eq.annual_output_kg or eq.annual_output_kg <= 0:
        return {
            "equipment_total": 0.0,
            "capex_total_musd": total_capex_musd,
            "annualized_capex_musd": annualized_capex_musd,
            "note": "annual_output_kg missing; equipment_total set to 0",
        }

    equipment_total = annualized_capex_usd / eq.annual_output_kg  # $/kg

    return {
        "equipment_total": equipment_total,
        "capex_total_musd": total_capex_musd,
        "annualized_capex_musd": annualized_capex_musd,
    }

# -----------------------------
# Orchestrator
# -----------------------------
def run_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    chem = Chemistry(**cfg.get("chemistry", {}))
    li = LiRoute(**cfg.get("lithium_route", {}))
    form = CathodeFormulation(**cfg.get("cathode_formulation", {}))
    prices = Prices(**cfg.get("prices", {}))
    ucons = UtilitiesConsumption(**cfg.get("utilities_consumption", {}))
    eq = EquipmentScaling(**cfg.get("equipment", {}))

    reag = reagent_masses_per_kg_cam(chem, li)
    extras = cathode_extras_per_kg_cam(form)

    mat_cost = materials_cost_per_kg(reag, extras, prices, li)
    uti_cost = utilities_cost_per_kg(ucons, prices)
    eq_cost = equipment_cost_per_kg(eq)

    total = mat_cost["materials_total"] + uti_cost["utilities_total"] + eq_cost["equipment_total"]

    return {
        "inputs": {
            "chemistry": asdict(chem),
            "lithium_route": asdict(li),
            "cathode_formulation": asdict(form),
            "prices": asdict(prices),
            "utilities_consumption": asdict(ucons),
            "equipment": {
                "baseline_musd": eq.baseline_musd,
                "exponents": eq.exponents,
                "S_ref_GWh": eq.S_ref_GWh,
                "S_target_GWh": eq.S_target_GWh,
                "life_years": eq.life_years,
                "annual_output_kg": eq.annual_output_kg,
            },
        },
        "computed": {
            "reagents_g_per_kg_cam": reag,
            "cathode_extras_g_per_kg_cam": extras,
        },
        "cost_breakdown_per_kg_cam": {
            "materials": mat_cost,
            "utilities": uti_cost,
            "equipment": eq_cost,
            "total_cost_per_kg_cam": total,
        },
    }

# -----------------------------
# Streamlit UI (tabs layout)
# -----------------------------
st.set_page_config(page_title="SC811 Cost Calculator", layout="wide")
st.title("SC811 Cost Calculator (NMC 811)")

# Sidebar: global toggles only
with st.sidebar:
    st.header("Global options")
    show_kpi_only = st.checkbox("Show total KPI only", value=False)
    lock_utils = st.checkbox("Lock utilities to WA prices", value=True)

# Default values (in case tabs are reordered later)
x_Ni = 0.8
y_Mn = 0.1
z_Co = 0.1
f_li = 1.02
u_li = 1.00
w_cam = 0.96
w_cb = 0.02
w_pvdf = 0.02
solids = 0.60
nmp_rec = 0.95

li2co3 = 96.0
lioh_h2o = 11.10
li2o_price = 29.15
naoh = 0.8
ni_price = 11.24
mn_price = 1.89
co_price = 38.24
cb_price = 4.0
pvdf_price = 20.0
nmp_price = 3.0

elec = 0.07
gas = 0.90
water = 0.0030
e_kwh = 1.5
g_therm = 0.02
w_gal = 0.1

S_ref = 50.0
S_tgt = 50.0
life = 10.0
specific_energy = 760.0
annual_kg = 0.0

labor_cost = 0.56
ware_cost = 0.09
maint_cost = 0.13
pack_cost = 0.17
reagent_cost = 1.07

eq_defaults = {
    "precursor_mixing": 230,
    "co_precipitation": 550,
    "solid_liquid_separation": 210,
    "drying": 360,
    "lithiation": 530,
    "milling": 260,
    "coating": 240,
    "environmental_controls": 410,
}

# Tabs
tab_mat, tab_eq, tab_util = st.tabs(
    ["Materials & Chemistry", "Equipment", "Utilities & Overhead"]
)

# -----------------------------
# Tab 1: Materials & Chemistry
# -----------------------------
with tab_mat:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Chemistry (NMC xyz)")
        x_Ni = st.number_input("Ni fraction (x)", 0.0, 1.0, 0.8, 0.01)
        y_Mn = st.number_input("Mn fraction (y)", 0.0, 1.0, 0.1, 0.01)
        z_Co = st.number_input("Co fraction (z)", 0.0, 1.0, 0.1, 0.01)
        st.caption("x + y + z should be close to 1.")

        st.subheader("Lithium factors")
        f_li = st.number_input("Li excess factor f_li", 1.0, 1.2, 1.02, 0.01)
        u_li = st.number_input("Li usage factor u_li", 1.0, 2.0, 1.00, 0.01)

    with colB:
        st.subheader("Material prices ($/kg)")
        li2co3 = st.number_input("Li2CO3", min_value=0.0, value=96.0, step=0.1)
        lioh_h2o = st.number_input("LiOH·H2O", min_value=0.0, value=11.10, step=0.1)
        li2o_price = st.number_input("Li2O (effective)", min_value=0.0, value=29.15, step=0.1)
        naoh = st.number_input("NaOH", min_value=0.0, value=0.8, step=0.1)

        ni_price = st.number_input("Ni (element)", min_value=0.0, value=11.24, step=0.01)
        mn_price = st.number_input("Mn (element)", min_value=0.0, value=1.89, step=0.01)
        co_price = st.number_input("Co (element)", min_value=0.0, value=38.24, step=0.01)

        cb_price = st.number_input("Carbon black (CB)", min_value=0.0, value=4.0, step=0.1)
        pvdf_price = st.number_input("PVDF", min_value=0.0, value=20.0, step=0.1)
        nmp_price = st.number_input("NMP", min_value=0.0, value=3.0, step=0.1)

    st.subheader("Cathode formulation")
    colC, colD, colE, colF = st.columns(4)
    with colC:
        w_cam = st.number_input("CAM fraction in solids", 0.5, 1.0, 0.96, 0.01)
    with colD:
        w_cb = st.number_input("CB fraction", 0.0, 0.3, 0.02, 0.005)
    with colE:
        w_pvdf = st.number_input("PVDF fraction", 0.0, 0.3, 0.02, 0.005)
    with colF:
        solids = st.number_input("Slurry solids loading", 0.1, 0.9, 0.60, 0.01)
    nmp_rec = st.number_input("NMP recovery", 0.0, 1.0, 0.95, 0.01)

# -----------------------------
# Tab 2: Equipment
# -----------------------------
with tab_eq:
    st.subheader("Equipment scaling")
    colE1, colE2 = st.columns(2)
    with colE1:
        S_ref = st.number_input("Baseline scale X (GWh/yr)", 3.6, 200.0, 50.0, 0.1)
        S_tgt = st.number_input("Target scale S (GWh/yr)", 3.6, 200.0, 50.0, 0.1)
        life = st.number_input("Equipment life (years)", min_value=1.0, value=10.0, step=1.0)
    with colE2:
        specific_energy = st.number_input("Cathode specific energy (Wh/kg)",
                                          min_value=100.0, value=760.0, step=10.0)
        st.caption("Cathode Mass (kg/yr)")

    annual_kg = (S_tgt * 1_000_000.0) / (specific_energy * 1_000.0)
    st.write(f"Computed cathode/CAM mass: **{annual_kg:,.2f} kg/yr**")

    st.subheader("Baseline CAPEX at X GWh (USD M)")
    df_eq = pd.DataFrame({
        "unit_op": list(eq_defaults.keys()),
        "baseline_$M": list(eq_defaults.values()),
        "exponent": [0.6] * len(eq_defaults),
    })
    df_eq = st.data_editor(
        df_eq,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# Tab 3: Utilities & Overhead
# -----------------------------
with tab_util:
    st.subheader("Utilities (WA prices or custom)")
    colU1, colU2 = st.columns(2)
    with colU1:
        elec = st.number_input("Electricity $/kWh",
                               min_value=0.0, value=0.07, step=0.01,
                               disabled=lock_utils)
        gas = st.number_input("Natural gas $/therm",
                              min_value=0.0, value=0.90, step=0.05,
                              disabled=lock_utils)
        water = st.number_input("Water $/gallon",
                                min_value=0.0, value=0.0030, step=0.0005,
                                disabled=lock_utils)
    with colU2:
        st.subheader("Consumption per kg CAM")
        e_kwh = st.number_input("Electricity kWh/kg", min_value=0.0, value=1.5, step=0.1)
        g_therm = st.number_input("Gas therm/kg", min_value=0.0, value=0.02, step=0.005)
        w_gal = st.number_input("Water gal/kg", min_value=0.0, value=0.1, step=0.01)

    st.subheader("Overhead (per kg CAM)")
    colO1, colO2, colO3, colO4, colO5 = st.columns(5)
    with colO1:
        labor_cost = st.number_input("Labor $/kg", min_value=0.0, value=0.56, step=0.01)
    with colO2:
        ware_cost = st.number_input("Warehousing $/kg", min_value=0.0, value=0.09, step=0.01)
    with colO3:
        maint_cost = st.number_input("Maintenance $/kg", min_value=0.0, value=0.13, step=0.01)
    with colO4:
        pack_cost = st.number_input("Packaging $/kg", min_value=0.0, value=0.17, step=0.01)
    with colO5:
        reagent_cost = st.number_input("Other reagents $/kg", min_value=0.0, value=1.07, step=0.01)

# -----------------------------
# Assemble config and run model
# -----------------------------
cfg = {
    "chemistry": {"x_Ni": x_Ni, "y_Mn": y_Mn, "z_Co": z_Co},
    "lithium_route": {"route": "hydroxide", "f_li": f_li, "u_li": u_li},
    "cathode_formulation": {
        "w_cam": w_cam,
        "w_cb": w_cb,
        "w_pvdf": w_pvdf,
        "solids_loading": solids,
        "nmp_recovery": nmp_rec,
    },
    "prices": {
        "li2co3": li2co3,
        "lioh_h2o": lioh_h2o,
        "li2o": li2o_price,
        "naoh": naoh,
        "ni_metal": ni_price,
        "mn_metal": mn_price,
        "co_metal": co_price,
        "cb": cb_price,
        "pvdf": pvdf_price,
        "nmp": nmp_price,
        "elec_per_kwh": 0.07 if lock_utils else elec,
        "gas_per_therm": 0.90 if lock_utils else gas,
        "water_per_gal": 0.0030 if lock_utils else water,
    },
    "utilities_consumption": {
        "electricity_kwh_per_kg": e_kwh,
        "gas_therm_per_kg": g_therm,
        "water_gal_per_kg": w_gal,
    },
    "equipment": {
        "baseline_musd": {row["unit_op"]: float(row["baseline_$M"]) for _, row in df_eq.iterrows()},
        "exponents": {row["unit_op"]: float(row["exponent"]) for _, row in df_eq.iterrows()},
        "S_ref_GWh": S_ref,
        "S_target_GWh": S_tgt,
        "life_years": life,
        "annual_output_kg": annual_kg,
    },
}

def build_tables(report, labor, ware, maint, pack, reag):
    mat = report["cost_breakdown_per_kg_cam"]["materials"]
    uti = report["cost_breakdown_per_kg_cam"]["utilities"]
    eqp = report["cost_breakdown_per_kg_cam"]["equipment"]

    li_cost = mat.get("Li_source", 0.0)
    ni_cost = mat.get("Ni", 0.0)
    mn_cost = mat.get("Mn", 0.0)
    co_cost = mat.get("Co", 0.0)
    coating_cost = mat.get("CB", 0.0) + mat.get("PVDF", 0.0) + mat.get("NMP_makeup", 0.0)

    ng_cost = uti.get("gas", 0.0)
    elec_cost = uti.get("electricity", 0.0)
    water_cost = uti.get("water", 0.0)

    equip_cost = eqp.get("equipment_total", 0.0)  # $/kg

    rows = [
        ("Li", li_cost),
        ("Ni", ni_cost),
        ("Mn", mn_cost),
        ("Co", co_cost),
        ("Coating", coating_cost),
        ("Natural Gas", ng_cost),
        ("Electricity", elec_cost),
        ("Water", water_cost),
        ("Labor", labor),
        ("Warehousing", ware),
        ("Maintenance", maint),
        ("Packaging", pack),
        ("Reagent", reag),
        ("Equipment", equip_cost),
    ]
    prod = sum(v for _, v in rows)
    rows.append(("Production Cost", prod))

    material_price = li_cost + ni_cost + mn_cost + co_cost + coating_cost + reag
    utility_price = ng_cost + elec_cost + water_cost
    overhead_price = labor + ware + maint + pack
    equipment_price = equip_cost

    return rows, material_price, utility_price, overhead_price, equipment_price, prod

# Run scenarios
cfg_lioh = json.loads(json.dumps(cfg))
cfg_lioh["lithium_route"]["route"] = "hydroxide"
rep_lioh = run_model(cfg_lioh)

cfg_li2o = json.loads(json.dumps(cfg))
cfg_li2o["lithium_route"]["route"] = "oxide"
rep_li2o = run_model(cfg_li2o)

# KPI
kpi_lioh = rep_lioh["cost_breakdown_per_kg_cam"]["total_cost_per_kg_cam"]
kpi_li2o = rep_li2o["cost_breakdown_per_kg_cam"]["total_cost_per_kg_cam"]

c_k1, c_k2 = st.columns(2)
with c_k1:
    st.metric("Total cost per kg CAM — LiOH", f"$ {kpi_lioh:.4f}")
with c_k2:
    st.metric("Total cost per kg CAM — Li₂O", f"$ {kpi_li2o:.4f}")

if show_kpi_only:
    st.stop()

# Equipment CAPEX summary (M$ / M$/yr)
eq_lioh = rep_lioh["cost_breakdown_per_kg_cam"]["equipment"]
eq_li2o = rep_li2o["cost_breakdown_per_kg_cam"]["equipment"]

capex_summary = pd.DataFrame({
    "Metric": ["Total CAPEX (M$)", "Annualized CAPEX (M$/yr)"],
    "LiOH": [eq_lioh.get("capex_total_musd", 0.0),
             eq_lioh.get("annualized_capex_musd", 0.0)],
    "Li2O": [eq_li2o.get("capex_total_musd", 0.0),
             eq_li2o.get("annualized_capex_musd", 0.0)],
}).round(3)

st.subheader("Equipment CAPEX Summary (M$ basis)")
st.dataframe(capex_summary, use_container_width=True)

# $/kg tables
rows_lioh, m_lioh, u_lioh, o_lioh, e_lioh, p_lioh = build_tables(
    rep_lioh, labor_cost, ware_cost, maint_cost, pack_cost, reagent_cost
)
rows_li2o, m_li2o, u_li2o, o_li2o, e_li2o, p_li2o = build_tables(
    rep_li2o, labor_cost, ware_cost, maint_cost, pack_cost, reagent_cost
)

df_components = pd.DataFrame({
    "Cost Component": [r[0] for r in rows_lioh],
    "LiOH": [round(r[1], 4) for r in rows_lioh],
    "Li2O": [round(r[1], 4) for r in rows_li2o],
})
st.subheader("NMC 811 — Component Breakdown ($/kg)")
st.dataframe(df_components, use_container_width=True)

df_agg = pd.DataFrame({
    "Category": ["Material", "Utility", "Overhead", "Equipment", "Production Cost"],
    "LiOH": [m_lioh, u_lioh, o_lioh, e_lioh, p_lioh],
    "Li2O": [m_li2o, u_li2o, o_li2o, e_li2o, p_li2o],
}).round(4)

st.subheader("Calculation Summary ($/kg basis)")
st.dataframe(df_agg, use_container_width=True)

# Pie charts
c1, c2 = st.columns(2)
with c1:
    st.write("**LiOH route cost share**")
    fig, ax = plt.subplots()
    ax.pie(
        [m_lioh, u_lioh, o_lioh, e_lioh],
        labels=["Material", "Utility", "Overhead", "Equipment"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")
    st.pyplot(fig)

with c2:
    st.write("**Li₂O route cost share**")
    fig, ax = plt.subplots()
    ax.pie(
        [m_li2o, u_li2o, o_li2o, e_li2o],
        labels=["Material", "Utility", "Overhead", "Equipment"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")
    st.pyplot(fig)