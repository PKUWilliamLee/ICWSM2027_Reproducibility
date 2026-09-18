"""Construct presentation tables and figure inputs from the current model fits."""
from pathlib import Path
import pandas as pd
import numpy as np
from verify_utils import ROOT, compare_frames, write_json
EMP=ROOT/'data/derived/empirical';OUT=ROOT/'outputs/tables';DER=ROOT/'outputs/derived/empirical'
THEMES=[('schoolwork_burden','Schoolwork Burden'),('shadow_education','Shadow Education'),('refund_disputes','Refund Disputes'),('after_school_services','After-school Services'),('family_pressure','Family Pressure'),('student_wellbeing','Student Well-being'),('admissions_equity','Admissions Equity')]
SPECS={'main_primary_share_aug2021':'table5_main_full_terms.csv','quadratic_primary_share_aug2021':'table6_quadratic_full_terms.csv','announcement_2021_07':'table7_july_full_terms.csv','delayed_2021_09':'table7_september_full_terms.csv','month_fe_primary_share_aug2021':'table8_month_effects_full_terms.csv','exclude_2025_primary_share':'table9_exclude2025_full_terms.csv','anylabel_share_aug2021':'table10_anylabel_full_terms.csv'}
def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True);DER.mkdir(parents=True,exist_ok=True)
    t=pd.read_csv(OUT/'itsa_all_specifications_terms.csv');e=pd.read_csv(OUT/'itsa_all_specifications_effects.csv')
    focal=t.loc[t.specification.eq('main_primary_share_aug2021') & t.term.isin(['time','post_policy','post_time'])].copy()
    focal.to_csv(OUT/'table2_main_itsa.csv',index=False)
    for spec,name in SPECS.items():
        block=t.loc[t.specification.eq(spec)].copy()
        # The fitted quadratic regressor is (time in months / 12)^2.
        # Present the coefficient and uncertainty in month-squared units.
        q=block.term.eq('time_sq')
        block.loc[q,['coefficient','std_error','ci_low','ci_high']] /= 144.0
        block.to_csv(OUT/name,index=False)
    e24=e.loc[e.horizon_months.eq(24),['specification','theme','effect_pp','ci_low_pp','ci_high_pp']].rename(columns={'effect_pp':'effect_24m_pp','ci_low_pp':'ci_low_24m_pp','ci_high_pp':'ci_high_24m_pp'})
    e24.to_csv(OUT/'table11_robustness_24m.csv',index=False)
    reference=pd.read_csv(EMP/'robustness_synthesis_24m.csv')
    check=compare_frames(e24,reference,['specification','theme'],columns=['effect_24m_pp','ci_low_24m_pp','ci_high_24m_pp'],atol=1e-6)
    plot=e.loc[e.specification.eq('main_primary_share_aug2021'),['theme','horizon_months','effect_pp','ci_low_pp','ci_high_pp']].rename(columns={'theme':'Theme','horizon_months':'horizon','effect_pp':'estimate','ci_low_pp':'low','ci_high_pp':'high'})
    plot.to_csv(DER/'itsa_counterfactual_effects_long.csv',index=False)
    monthly=pd.read_csv(EMP/'monthly_theme_panel.csv',parse_dates=['ym']);monthly['year']=monthly.ym.dt.year
    grouped=monthly.groupby('year',sort=True).sum(numeric_only=True)
    annual=pd.DataFrame({'year':grouped.index})
    for slug,name in THEMES:annual[name]=(grouped['primary_count_'+slug]/grouped.total_messages).to_numpy()
    annual.to_csv(DER/'annual_primary_theme_shares.csv',index=False)
    annual_check=compare_frames(annual,pd.read_csv(EMP/'annual_primary_theme_shares.csv'),['year'],atol=1e-12,rtol=0)
    for name in ['cooccurrence_selected_edges.csv','cooccurrence_all_21_edges.csv']:
        frame=pd.read_csv(EMP/name);frame.to_csv(DER/name,index=False)
    pd.read_csv(EMP/'cooccurrence_all_21_edges.csv').to_csv(OUT/'table12_cooccurrence.csv',index=False)
    write_json(ROOT/'outputs/empirical_table_checks.json',{'annual_from_monthly_counts':annual_check,'table11':check,'cooccurrence_starting_level':'Released aggregated edge shares; no message-level recount'})
    print('Tables 2 and 5-12 prepared; empirical figure inputs use the current fit.')
if __name__=='__main__':main()
