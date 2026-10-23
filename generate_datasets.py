import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Women Safety Geospatial Dataset
def generate_women_safety_dataset():
    np.random.seed(42)
    cities_coords = {
        "Delhi": (28.6139, 77.2090, [
            ("Connaught Place", 28.6315, 77.2167),
            ("Hauz Khas", 28.5494, 77.2001),
            ("Chandni Chowk", 28.6506, 77.2303),
            ("Saket", 28.5244, 77.2173),
            ("Rohini Sector 10", 28.7159, 77.1190),
            ("Dwarka Sector 21", 28.5524, 77.0583),
            ("Lajpat Nagar", 28.5677, 77.2433),
            ("Karol Bagh", 28.6514, 77.1907),
            ("Janakpuri", 28.6219, 77.0878),
            ("Mayur Vihar", 28.6095, 77.2982),
        ]),
        "Mumbai": (19.0760, 72.8777, [
            ("Bandra West", 19.0596, 72.8295),
            ("Andheri East", 19.1136, 72.8697),
            ("Colaba", 18.9067, 72.8147),
            ("Dadar", 19.0178, 72.8478),
            ("Juhu Beach", 19.0988, 72.8264),
            ("Powai", 19.1176, 72.9060),
            ("Kurla West", 19.0657, 72.8794),
            ("Goregaon Film City", 19.1663, 72.8526),
            ("Marine Drive", 18.9432, 72.8230),
            ("Borivali West", 19.2307, 72.8567),
        ]),
        "Bengaluru": (12.9716, 77.5946, [
            ("Koramangala 5th Block", 12.9352, 77.6245),
            ("Indiranagar 100ft Rd", 12.9784, 77.6408),
            ("Whitefield ITPL", 12.9855, 77.7317),
            ("Electronic City Phase 1", 12.8452, 77.6602),
            ("MG Road Metro", 12.9756, 77.6066),
            ("HSR Layout Sector 1", 12.9121, 77.6446),
            ("Jayanagar 4th Block", 12.9299, 77.5824),
            ("Hebbal Flyover", 13.0358, 77.5970),
            ("Marathahalli Bridge", 12.9591, 77.6974),
            ("Malleshwaram", 13.0031, 77.5643),
        ]),
        "Kolkata": (22.5726, 88.3639, [
            ("Park Street", 22.5511, 88.3524),
            ("Salt Lake Sector V", 22.5804, 88.4378),
            ("Howrah Station Area", 22.5830, 88.3426),
            ("New Town Action Area 1", 22.5867, 88.4623),
            ("Gariahat Market", 22.5186, 88.3653),
            ("Shyambazar Five Point", 22.6021, 88.3712),
            ("Ballygunge Circular Rd", 22.5312, 88.3582),
            ("Esplanade Metro", 22.5645, 88.3512),
        ]),
        "Chennai": (13.0827, 80.2707, [
            ("T. Nagar Pondy Bazaar", 13.0418, 80.2337),
            ("Marina Beach Promenade", 13.0500, 80.2824),
            ("Adyar Kasturba Nagar", 13.0012, 80.2565),
            ("Velachery Main Rd", 12.9759, 80.2212),
            ("Anna Nagar Roundtana", 13.0850, 80.2101),
            ("OMR Thoraipakkam", 12.9382, 80.2323),
            ("Guindy Kathipara", 13.0067, 80.2038),
            ("Egmore Railway Station", 13.0784, 80.2612),
        ]),
        "Hyderabad": (17.3850, 78.4867, [
            ("Hitec City Cyber Towers", 17.4504, 78.3808),
            ("Banjara Hills Rd 12", 17.4156, 78.4350),
            ("Gachibowli Stadium", 17.4401, 78.3489),
            ("Charminar Old City", 17.3616, 78.4747),
            ("Jubilee Hills Checkpost", 17.4319, 78.4073),
            ("Secunderabad Clock Tower", 17.4399, 78.4983),
            ("Begumpet Airport Rd", 17.4448, 78.4682),
            ("Kukatpally Housing Board", 17.4938, 78.3980),
        ]),
        "Pune": (18.5204, 73.8567, [
            ("FC Road Deccan", 18.5203, 73.8409),
            ("Koregaon Park North Main Rd", 18.5362, 73.8940),
            ("Hinjawadi Phase 1 IT", 18.5913, 73.7389),
            ("Viman Nagar Phoenix", 18.5620, 73.9168),
            ("Kothrud Paud Road", 18.5074, 73.8077),
            ("Magarpatta Cybercity", 18.5147, 73.9298),
            ("Swargate Bus Stand", 18.5018, 73.8586),
        ]),
        "Jaipur": (26.9124, 75.7873, [
            ("MI Road Panch Batti", 26.9165, 75.8076),
            ("Malviya Nagar GT Central", 26.8530, 75.8052),
            ("Raja Park Fashion St", 26.8997, 75.8284),
            ("Vaishali Nagar Amrapali", 26.9114, 75.7431),
            ("C-Scheme Subhash Marg", 26.9067, 75.8012),
        ])
    }

    rows = []
    idx = 1001
    for city, (base_lat, base_lon, spots) in cities_coords.items():
        for spot_name, lat, lon in spots:
            for rep in range(4):  # Multiple observation windows (Day, Dusk, Night, Late Night)
                time_of_day = ["Day", "Dusk", "Night", "Late Night"][rep]
                jitter_lat = lat + np.random.normal(0, 0.003)
                jitter_lon = lon + np.random.normal(0, 0.003)

                lighting_opts = ["Good", "Moderate", "Poor"]
                lighting_p = [0.6, 0.3, 0.1] if time_of_day == "Day" else [0.2, 0.4, 0.4] if time_of_day == "Late Night" else [0.35, 0.45, 0.2]
                lighting = np.random.choice(lighting_opts, p=lighting_p)

                patrol_opts = ["High", "Moderate", "Low"]
                patrol_p = [0.4, 0.4, 0.2] if "Metro" in spot_name or "Police" in spot_name or "CP" in spot_name else [0.2, 0.45, 0.35]
                patrol = np.random.choice(patrol_opts, p=patrol_p)

                crowd = np.random.choice(["High", "Medium", "Low"], p=[0.5, 0.35, 0.15] if time_of_day in ["Day", "Dusk"] else [0.1, 0.3, 0.6])
                cctv = np.random.choice(["Yes", "Partial", "No"], p=[0.45, 0.35, 0.20])
                transport = np.random.choice(["High", "Moderate", "Low"], p=[0.5, 0.35, 0.15])
                dist_emergency = round(float(np.random.gamma(shape=2.5, scale=1.2)), 2)
                incidents = int(np.random.poisson(lam=4.2 if lighting == "Poor" and patrol == "Low" else 1.8))

                # Compute synthetic safety index (1.0 to 10.0 scale)
                base_score = 6.8
                if lighting == "Good": base_score += 1.4
                elif lighting == "Poor": base_score -= 1.8
                if patrol == "High": base_score += 1.2
                elif patrol == "Low": base_score -= 1.2
                if cctv == "Yes": base_score += 0.8
                elif cctv == "No": base_score -= 0.8
                if transport == "High": base_score += 0.5
                if time_of_day == "Late Night": base_score -= 0.9
                base_score -= min(incidents * 0.25, 2.0)
                base_score += np.random.normal(0, 0.35)
                safety_score = round(float(np.clip(base_score, 1.2, 9.8)), 1)

                risk_level = "Safe (Low Risk)" if safety_score >= 7.0 else "Moderate Risk" if safety_score >= 4.5 else "High Risk Hazard"

                rows.append({
                    "incident_id": f"WSG-{idx}",
                    "city": city,
                    "area_name": spot_name,
                    "latitude": round(jitter_lat, 6),
                    "longitude": round(jitter_lon, 6),
                    "time_of_day": time_of_day,
                    "safety_score": safety_score,
                    "risk_category": risk_level,
                    "lighting_quality": lighting,
                    "police_patrol_frequency": patrol,
                    "crowd_density": crowd,
                    "cctv_coverage": cctv,
                    "emergency_services_proximity_km": dist_emergency,
                    "public_transport_access": transport,
                    "reported_incidents_past_year": incidents
                })
                idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "women_safety_geospatial.csv", index=False)
    print(f"Generated {len(df)} rows in data/women_safety_geospatial.csv")

# 2. Bharatiya Nyaya Sanhita (BNS) 2023 Dataset
def generate_bns_dataset():
    bns_data = [
        {
            "section": "Section 63",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Rape (Definition)",
            "punishment": "Defined under Section 64 (Rigorous imprisonment not less than 10 years extending to life)",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 375",
            "description": "A man is said to commit rape who penetrates or inserts objects into woman's body or mouth, or applies mouth to genitalia, without consent, under fear of death/hurt, false belief of marriage, unsoundness of mind, intoxication, or under 18 years of age."
        },
        {
            "section": "Section 64",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Punishment for rape",
            "punishment": "Rigorous imprisonment of not less than 10 years up to imprisonment for life, and fine (payable to victim)",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 376(1)",
            "description": "Whoever commits rape shall be punished with rigorous imprisonment of either description for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine. Fine shall be just and reasonable to meet medical expenses."
        },
        {
            "section": "Section 65",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Punishment for rape in certain cases (Woman under 16 / 12 years)",
            "punishment": "Rape of woman under 16: RI not less than 20 years to life; Under 12: RI not less than 20 years to life or death",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 376(3), 376AB",
            "description": "Rape committed on a woman under sixteen years of age is punishable with rigorous imprisonment not less than twenty years to life imprisonment. If the victim is under twelve years, punishment extends to death."
        },
        {
            "section": "Section 66",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Rape resulting in death or persistent vegetative state of victim",
            "punishment": "Rigorous imprisonment of not less than 20 years extending to life (remainder of natural life) or with death",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 376A",
            "description": "Whoever inflicts an injury in the course of rape which causes death or causes the woman to be in a persistent vegetative state shall be punished with imprisonment for not less than twenty years up to life, or death."
        },
        {
            "section": "Section 67",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Sexual intercourse by person in authority",
            "punishment": "Rigorous imprisonment of not less than 5 years extending to 10 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 376C",
            "description": "Whoever being in a position of authority or in a fiduciary relationship, or being a superintendent of remand home, hospital, or educational institution abuses position to induce or seduce any woman."
        },
        {
            "section": "Section 68",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Sexual intercourse by husband upon his wife during separation",
            "punishment": "Imprisonment of not less than 2 years extending up to 7 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 376B",
            "description": "Whoever has sexual intercourse with his wife living separately under a decree of separation or under any custom or usage, without her consent, is punishable under this section."
        },
        {
            "section": "Section 69",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Sexual intercourse by employing deceitful means (False promise to marry)",
            "punishment": "Imprisonment of either description for a term which may extend to 10 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Newly introduced in BNS 2023",
            "description": "Whoever, by deceitful means or by making promise to marry a woman without any intention of fulfilling the same, has sexual intercourse with her, not amounting to the offence of rape. Deceitful means include false promise of employment, promotion, or marrying after suppressing identity."
        },
        {
            "section": "Section 70",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Gang rape",
            "punishment": "RI not less than 20 years extending to life (natural life), and fine; If victim under 18: life (natural life) or death",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 376D, 376DA, 376DB",
            "description": "Where a woman is raped by one or more persons constituting a group or acting in furtherance of a common intention, each of those persons shall be deemed to have committed the offence of gang rape."
        },
        {
            "section": "Section 71",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Repeat offenders of rape",
            "punishment": "Imprisonment for life (natural life) or with death",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 376E",
            "description": "Whoever having been convicted of an offence punishable under section 64 or section 65 or section 70 is subsequently convicted of an offence under any of those sections shall be punished with life or death."
        },
        {
            "section": "Section 72",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Disclosure of identity of victim of certain offences (Rape)",
            "punishment": "Imprisonment of either description for a term which may extend to 2 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Bailable",
            "triable_by": "Magistrate of First Class",
            "ipc_corresponding": "Section 228A",
            "description": "Whoever prints or publishes the name or any matter which may make known the identity of any person against whom an offence under Section 64/65/66/67/68/69/70/71 is alleged to have been committed."
        },
        {
            "section": "Section 74",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Assault or criminal force to woman with intent to outrage her modesty",
            "punishment": "Imprisonment not less than 1 year extending to 5 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Any Magistrate",
            "ipc_corresponding": "Section 354",
            "description": "Whoever assaults or uses criminal force to any woman, intending to outrage or knowing it to be likely that he will thereby outrage her modesty, shall be punished."
        },
        {
            "section": "Section 75",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Sexual harassment",
            "punishment": "Physical contact: up to 3 years; Demanding sexual favours/showing pornography/lewd remarks: up to 1-3 years and fine",
            "cognizable": "Cognizable",
            "bailable": "Bailable (1st offence)",
            "triable_by": "Any Magistrate",
            "ipc_corresponding": "Section 354A",
            "description": "A man committing unwelcome physical contact, demand or request for sexual favours, showing pornography against the will of a woman, or making sexually coloured remarks shall be guilty of sexual harassment."
        },
        {
            "section": "Section 76",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Assault or use of criminal force to woman with intent to disrobe",
            "punishment": "Imprisonment of not less than 3 years extending to 7 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 354B",
            "description": "Any man who assaults or uses criminal force to any woman or abets such act with the intention of disrobing or compelling her to be naked."
        },
        {
            "section": "Section 77",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Voyeurism",
            "punishment": "First conviction: 1 to 3 years and fine; Second/subsequent conviction: 3 to 7 years and fine",
            "cognizable": "Cognizable",
            "bailable": "Bailable (1st offence), Non-Bailable (subsequent)",
            "triable_by": "Any Magistrate",
            "ipc_corresponding": "Section 354C",
            "description": "Any man who watches, or captures the image of a woman engaging in a private act in circumstances where she would usually have the expectation of not being observed, or disseminates such image."
        },
        {
            "section": "Section 78",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Stalking",
            "punishment": "First conviction: up to 3 years and fine; Second/subsequent conviction: up to 5 years and fine",
            "cognizable": "Cognizable",
            "bailable": "Bailable (1st offence), Non-Bailable (subsequent)",
            "triable_by": "Any Magistrate",
            "ipc_corresponding": "Section 354D",
            "description": "Any man who follows a woman and contacts, or attempts to contact such woman repeatedly despite clear indication of disinterest, or monitors the use by a woman of the internet, email or other electronic communication."
        },
        {
            "section": "Section 79",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Word, gesture or act intended to insult the modesty of a woman",
            "punishment": "Simple imprisonment for a term which may extend to 3 years, and with fine",
            "cognizable": "Cognizable",
            "bailable": "Bailable",
            "triable_by": "Any Magistrate",
            "ipc_corresponding": "Section 509",
            "description": "Whoever, intending to insult the modesty of any woman, utters any word, makes any sound or gesture, or exhibits any object, intending that such word or sound shall be heard, or that such gesture or object shall be seen, or intrudes upon her privacy."
        },
        {
            "section": "Section 80",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Dowry death",
            "punishment": "Imprisonment for a term which shall not be less than 7 years but which may extend to imprisonment for life",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 304B",
            "description": "Where the death of a woman is caused by any burns or bodily injury or occurs otherwise than under normal circumstances within seven years of her marriage, and it is shown that soon before her death she was subjected to cruelty or harassment by her husband or his relatives in connection with dowry."
        },
        {
            "section": "Section 85",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Cohabitation caused by a man deceitfully inducing a belief of lawful marriage",
            "punishment": "Imprisonment of either description for a term which may extend to 10 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 493",
            "description": "Every man who by deceit causes any woman who is not lawfully married to him to believe that she is lawfully married to him and to cohabit or have sexual intercourse with him in that belief."
        },
        {
            "section": "Section 86",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Cruelty by husband or relatives of husband (Domestic Violence)",
            "punishment": "Imprisonment for a term which may extend to 3 years, and shall also be liable to fine",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Magistrate of First Class",
            "ipc_corresponding": "Section 498A",
            "description": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty. Cruelty means conduct likely to drive the woman to suicide or cause grave injury, or harassment to coerce unlawful property/dowry demands."
        },
        {
            "section": "Section 87",
            "chapter": "Chapter V: Of Offences Against Women and Children",
            "offence": "Kidnapping, abducting or inducing woman to compel her marriage",
            "punishment": "Imprisonment of either description for a term which may extend to 10 years, and fine",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 366",
            "description": "Whoever kidnaps or abducts any woman with intent that she may be compelled, or knowing it to be likely that she will be compelled, to marry any person against her will, or forced or seduced to illicit intercourse."
        },
        {
            "section": "Section 103",
            "chapter": "Chapter VI: Of Offences Affecting the Human Body",
            "offence": "Punishment for murder",
            "punishment": "Death or imprisonment for life, and shall also be liable to fine (Mob lynching specified in clause 2)",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 302",
            "description": "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine. When murder is committed on grounds of race, caste, community, sex or place of birth by group of 5 or more, each person is punished with death or life imprisonment."
        },
        {
            "section": "Section 109",
            "chapter": "Chapter VI: Of Offences Affecting the Human Body",
            "offence": "Attempt to murder",
            "punishment": "Imprisonment up to 10 years and fine; if hurt caused, life imprisonment or up to 10 years",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 307",
            "description": "Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused death, he would be guilty of murder."
        },
        {
            "section": "Section 115",
            "chapter": "Chapter VI: Of Offences Affecting the Human Body",
            "offence": "Voluntarily causing hurt",
            "punishment": "Imprisonment of either description for a term which may extend to 1 year, or with fine up to Rs. 10,000, or both",
            "cognizable": "Non-Cognizable",
            "bailable": "Bailable",
            "triable_by": "Any Magistrate",
            "ipc_corresponding": "Section 323",
            "description": "Whoever does any act with the intention of thereby causing hurt to any person, or with the knowledge that he is likely thereby to cause hurt to any person."
        },
        {
            "section": "Section 124",
            "chapter": "Chapter VI: Of Offences Affecting the Human Body",
            "offence": "Voluntarily causing grievous hurt by use of acid, etc.",
            "punishment": "Imprisonment of not less than 10 years extending to life imprisonment, and fine payable to victim",
            "cognizable": "Cognizable",
            "bailable": "Non-Bailable",
            "triable_by": "Court of Session",
            "ipc_corresponding": "Section 326A",
            "description": "Whoever causes permanent or partial damage or deformity to, or bums or maims or disfigures or disables, any part of the body of a person or causes grievous hurt by administering acid. Fine must cover victim treatment."
        },
        {
            "section": "Section 351",
            "chapter": "Chapter XVIII: Of Criminal Intimidation, Insult, Annoyance",
            "offence": "Criminal intimidation",
            "punishment": "Imprisonment up to 2 years, or fine, or both; if threat to cause death/grievous hurt: up to 7 years",
            "cognizable": "Non-Cognizable",
            "bailable": "Bailable",
            "triable_by": "Any Magistrate",
            "ipc_corresponding": "Section 506",
            "description": "Whoever threatens another with any injury to his person, reputation or property, or to the person or reputation of any one in whom that person is interested, with intent to cause alarm."
        }
    ]
    df = pd.DataFrame(bns_data)
    df.to_csv(DATA_DIR / "bns_2023_sections.csv", index=False)
    print(f"Generated {len(df)} rows in data/bns_2023_sections.csv")

# 3. Longitudinal Psychometric Counselling Evaluation Dataset
def generate_psychometric_dataset(n_participants=500):
    np.random.seed(42)
    pids = [f"PID_{i:04d}" for i in range(1, n_participants + 1)]
    ages = np.random.randint(18, 55, size=n_participants)
    
    # Latent distress factor at baseline ~ Normal(0, 1)
    latent_distress_t0 = np.random.normal(0.6, 0.9, size=n_participants)
    
    # Intervention sessions attended (4 to 12 sessions)
    sessions = np.random.randint(4, 13, size=n_participants)
    # Therapeutic response depends on sessions and baseline distress
    therapeutic_gain = 0.35 * (sessions / 8.0) + np.random.normal(0.4, 0.25, size=n_participants)
    latent_distress_t1 = latent_distress_t0 - therapeutic_gain

    records = []
    for i in range(n_participants):
        pid = pids[i]
        age = int(ages[i])
        sess = int(sessions[i])

        # T0 Baseline
        d0 = latent_distress_t0[i]
        # PHQ-9 items (0 to 3 scale)
        phq_t0 = [int(np.clip(np.round(1.5 + 0.65 * d0 + np.random.normal(0, 0.5)), 0, 3)) for _ in range(9)]
        # GAD-7 items (0 to 3 scale)
        gad_t0 = [int(np.clip(np.round(1.4 + 0.60 * d0 + np.random.normal(0, 0.5)), 0, 3)) for _ in range(7)]
        # Brief COPE Coping items (1 to 5 scale, negatively related to distress)
        coping_t0 = [int(np.clip(np.round(3.0 - 0.45 * d0 + np.random.normal(0, 0.6)), 1, 5)) for _ in range(6)]
        # Generic items item_1 to item_6 (1 to 5 scale) for backward compatibility
        item_t0 = [int(np.clip(np.round(3.2 + 0.60 * d0 + np.random.normal(0, 0.6)), 1, 5)) for _ in range(6)]

        rec_t0 = {
            "participant_id": pid,
            "timepoint": "T0",
            "age": age,
            "counselling_sessions": 0,
            "baseline_stress_index": round(float(2.5 + d0), 2),
            "phq_total": sum(phq_t0),
            "gad_total": sum(gad_t0),
            "coping_total": sum(coping_t0),
            "distress_severity": "Severe" if sum(phq_t0) >= 15 else "Moderate" if sum(phq_t0) >= 10 else "Mild"
        }
        for j, val in enumerate(phq_t0, 1): rec_t0[f"phq_{j}"] = val
        for j, val in enumerate(gad_t0, 1): rec_t0[f"gad_{j}"] = val
        for j, val in enumerate(coping_t0, 1): rec_t0[f"coping_{j}"] = val
        for j, val in enumerate(item_t0, 1): rec_t0[f"item_{j}"] = val
        records.append(rec_t0)

        # T1 Post-Counselling
        d1 = latent_distress_t1[i]
        phq_t1 = [int(np.clip(np.round(1.5 + 0.65 * d1 + np.random.normal(0, 0.5)), 0, 3)) for _ in range(9)]
        gad_t1 = [int(np.clip(np.round(1.4 + 0.60 * d1 + np.random.normal(0, 0.5)), 0, 3)) for _ in range(7)]
        coping_t1 = [int(np.clip(np.round(3.0 - 0.45 * d1 + np.random.normal(0, 0.6)), 1, 5)) for _ in range(6)]
        item_t1 = [int(np.clip(np.round(3.2 + 0.60 * d1 + np.random.normal(0, 0.6)), 1, 5)) for _ in range(6)]

        phq_diff = sum(phq_t0) - sum(phq_t1)
        rec_t1 = {
            "participant_id": pid,
            "timepoint": "T1",
            "age": age,
            "counselling_sessions": sess,
            "baseline_stress_index": round(float(2.5 + d0), 2),
            "phq_total": sum(phq_t1),
            "gad_total": sum(gad_t1),
            "coping_total": sum(coping_t1),
            "distress_severity": "Severe" if sum(phq_t1) >= 15 else "Moderate" if sum(phq_t1) >= 10 else "Mild",
            "clinical_responder": 1 if phq_diff >= 4 else 0
        }
        for j, val in enumerate(phq_t1, 1): rec_t1[f"phq_{j}"] = val
        for j, val in enumerate(gad_t1, 1): rec_t1[f"gad_{j}"] = val
        for j, val in enumerate(coping_t1, 1): rec_t1[f"coping_{j}"] = val
        for j, val in enumerate(item_t1, 1): rec_t1[f"item_{j}"] = val
        records.append(rec_t1)

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "psychometric_longitudinal_counselling.csv", index=False)
    print(f"Generated {len(df)} rows in data/psychometric_longitudinal_counselling.csv")

if __name__ == "__main__":
    generate_women_safety_dataset()
    generate_bns_dataset()
    generate_psychometric_dataset()
