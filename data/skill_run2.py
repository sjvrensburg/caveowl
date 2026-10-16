"""Run-2 skill trajectory prompts: pdf-explore, humanizer and coffee, the families where run 1
regressed on eval/skill_gate_wide.py (README "Run 1": humanizer stock 6/9 -> caveowl 2/9,
pdf-explore 9/9 -> 6/9, every loss `skill_not_read`: caveowl does the task directly and never
opens the skill's SKILL.md), plus coffee, which produced 0 kept rows in run 1.

Why coffee had 0 rows: not the model. Stock read SKILL.md and ran report.py in 26 of 30 draws
and judge_session passed them, but every one was dropped as `too_long` at --max-len 4096:
coffee-price-tracker's SKILL.md alone is 13 KB (~3.3k tokens), so SKILL.md + pi's trimmed system
prompt (~1.9k) + report.py's output is ~6-7k tokens before the model says a word. humanizer's
SKILL.md is 29 KB (~7k tokens), so its rows are longer still. These families cannot fit 4096 at
all; run 2 is converted with a larger --max-len and the operator decides the training
max_seq_length (owl_data drops rows longer than it). The prompts here also use the gate's
stricter coffee judging (`ground_alt` + `script_failed`, skill_gate_wide.judge), not run 1's
`ground: number`, which the gate found passes on stray numbers.

Each family mirrors how eval/skill_gate_wide.py defines it (same skill, same kind of fixture,
same success judge: skill_gate_wide.judge, selected by spec["judge"] == "wide"), but prompts
AND fixtures are disjoint from the gate's, so the gate stays a held-out test:
  - pdf-explore: nine locally generated multi-page "papers" (topics, file names, identifiers and
    numbers of their own), three questions each whose answer needs two pages;
  - humanizer: 26 AI-sounding paragraphs written for this file, one file each, rewritten in
    file mode; a distinctive number must survive (the skill forbids changing what text says);
  - coffee: 25 new questions answered from report.py over the tracker's real SQLite (read-only;
    the bench guard refuses scan_specials/setup_address).
check_disjoint() enforces it: no prompt equal to a gate, skill_gate or run-1 prompt; no fixture
file name the gate uses; no gate answer value (2+ characters) appearing as a token in a run-2
fixture; no run-2 answer value in a same-family gate fixture; no 8-word run shared with a gate fixture text.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

PDF_SKILLS = ["ocr-and-documents", "pdf-explore", "summarize"]
PDF_SCRIPT_RE = r"pdftotext|extract_pymupdf|fitz|pymupdf"
HUMANIZER_SKILLS = ["humanizer", "sepia", "academic-humanizer"]
RETAILERS = ["Checkers", "Pick n Pay", "Takealot"]


# --------------------------------------------------------------------------- pdf-explore

def _pdf_docs(banned: set[str]) -> list[tuple[str, list[list[str]], list[tuple[str, dict]]]]:
    """(file name, pages, [(question, expect spec)]). Values are seeded and re-rolled if they
    collide with a gate answer value."""
    r = random.Random("caveowl-run2-pdf-explore")

    def num(lo: int, hi: int) -> str:
        while True:
            v = str(r.randint(lo, hi))
            if v not in banned:
                return v

    def pct(lo: float, hi: float, dp: int = 1) -> str:
        while True:
            v = f"{r.uniform(lo, hi):.{dp}f}"
            if v not in banned:
                return v

    def pair(lo: float, hi: float, dp: int = 1) -> tuple[str, str]:
        """(abstract value, table value): equal about half the time, else a different value."""
        tab = pct(lo, hi, dp)
        if r.random() < 0.5:
            return tab, tab
        while True:
            ab = pct(lo, hi, dp)
            if ab != tab:
                return ab, tab

    docs = []

    # A: arsenic in boreholes
    n, k, z = num(140, 420), num(4, 9), num(7000000, 7999999)
    ab, tab = pair(9, 31)
    docs.append(("arsenic_wells.pdf", [
        ["Arsenic in rural boreholes of the Northern Cape", "Abstract",
         f"We tested drinking-water boreholes and found that {ab}% exceeded the 10 ug/L guideline.",
         "The sampling design is described on page 2 and the results on page 3."],
        ["2. Methods", f"Field teams sampled {n} boreholes in {k} districts between March and June.",
         "Water was filtered on site and analysed by ICP-MS within 14 days.",
         f"Data availability: all measurements are archived at Zenodo, DOI 10.5281/zenodo.{z}."],
        ["3. Results", "Table 1. Share of boreholes above 10 ug/L by district",
         f"Namakwa {pct(5, 40)}%   ZF Mgcawu {pct(5, 40)}%   Pixley ka Seme {pct(5, 40)}%",
         f"Overall exceedance: {tab}%", "Figure 2 maps every borehole above the guideline."]],
        [("arsenic_wells.pdf: how many boreholes were sampled, and what overall exceedance does the results table give?",
          {"expect_all_norm": [n, tab]}),
         ("Check whether the abstract of arsenic_wells.pdf agrees with its own results table on the exceedance rate.",
          {"expect_all_norm": [ab, tab]}),
         ("What DOI does arsenic_wells.pdf give for its data, and which page is it on?",
          {"expect_norm": [f"zenodo.{z}"]})]))

    # B: malaria rapid diagnostic test (4 pages, supplement)
    n, k, nct = num(310, 980), num(3, 8), f"NCT0{num(5000000, 6999999)}"
    sens, spec, bad = pct(88, 97), pct(90, 99), num(11, 39)
    docs.append(("rdt_trial.pdf", [
        ["Field accuracy of a malaria rapid diagnostic test in Mozambique", "Abstract",
         "We evaluated a two-band rapid test against a molecular reference in primary clinics.",
         "Methods: page 2. Results: page 3. Supplementary tables: page 4."],
        ["2. Methods", f"We enrolled {n} febrile patients at {k} primary clinics.",
         "Each patient gave one finger-prick sample; the reference standard was PCR.",
         f"Trial registration: {nct}."],
        ["3. Results", "Table 2. Test performance by clinic",
         f"Pooled sensitivity: {sens}%", f"Specificity: {spec}%",
         "Figure 1 shows the patient flow diagram."],
        ["Supplement", f"Table S1 lists the {bad} invalid test strips excluded from analysis.",
         "Table S2 gives sensitivity by parasite density band."]],
        [("In rdt_trial.pdf, what was the pooled sensitivity and what reference standard was it measured against?",
          {"expect_all_norm": [sens, "PCR"]}),
         ("What's the trial registration number in rdt_trial.pdf, and how many patients were enrolled?",
          {"expect_all_norm": [nct, n]}),
         ("How many invalid strips does rdt_trial.pdf's supplement say were excluded, and what specificity do the results report?",
          {"expect_all_norm": [bad, spec]})]))

    # C: wind turbine fault detection
    n, months, oedp = num(23, 88), num(14, 36), f"OEDP-{num(10000, 99999)}"
    f_ab, f_tab = pair(0.71, 0.93, dp=2)
    faults = {"gearbox": f"0.{num(60, 79)}", "generator": f"0.{num(80, 95)}", "pitch system": f"0.{num(50, 69)}"}
    best = max(faults, key=lambda x: float(faults[x]))
    docs.append(("turbine_faults.pdf", [
        ["Early fault detection for onshore wind turbines from SCADA data", "Abstract",
         f"A gradient-boosted classifier detects faults days ahead with an F1 of {f_ab}.",
         "Data and methods are on page 2; per-fault results are in Table 3 on page 3."],
        ["2. Methods", f"We used ten-minute SCADA records from {n} turbines over {months} months.",
         "Labels came from the operator's maintenance logs.",
         f"Data: Open Energy Data Portal record {oedp}."],
        ["3. Results", "Table 3. F1 by fault type",
         "   ".join(f"{k2} {v}" for k2, v in faults.items()),
         f"Macro F1: {f_tab}", "Most false alarms fell in the first week after maintenance."]],
        [("turbine_faults.pdf: across how many turbines was the model trained, and what macro F1 does Table 3 report?",
          {"expect_all_norm": [n, f_tab]}),
         ("Which fault type scores best in turbine_faults.pdf's results, and with what F1?",
          {"expect_all_norm": [best, faults[best]]}),
         ("Does the F1 quoted in the abstract of turbine_faults.pdf match the macro F1 in the results?",
          {"expect_all_norm": [f_ab, f_tab]})]))

    # D: drought-tolerant maize
    plots, sites, cml = num(96, 240), num(4, 7), f"CML{num(500, 599)}"
    y1, y2 = pct(4.1, 6.4), pct(2.2, 3.9)
    adv_ab, adv_tab = pair(18, 44)
    docs.append(("maize_trial.pdf", [
        ["Drought-tolerant maize under farmer management in Limpopo", "Abstract",
         f"The drought-tolerant hybrid out-yielded the conventional check by {adv_ab}%.",
         "Design on page 2, yields on page 3."],
        ["2. Methods", f"We planted {plots} plots across {sites} sites in a randomised block design.",
         f"Germplasm: the drought-tolerant entry is CIMMYT line {cml}.",
         "Rainfall was logged daily at each site."],
        ["3. Results", "Table 4. Grain yield (t/ha)",
         f"Drought-tolerant hybrid {y1}   Conventional check {y2}",
         f"Yield advantage: {adv_tab}%", "Two sites lost plots to hail and were kept in the analysis."]],
        [("What grain yields does maize_trial.pdf report for the two entries, and how many plots were planted?",
          {"expect_all_norm": [y1, y2, plots]}),
         ("Which germplasm line was tested in maize_trial.pdf, and on which page is that stated?",
          {"expect_norm": [cml]}),
         ("Is the yield advantage claimed in maize_trial.pdf's abstract the same as the one in the results?",
          {"expect_all_norm": [adv_ab, adv_tab]})]))

    # E: school dental survey
    n, schools, hrec = num(400, 900), num(6, 15), f"HREC {num(100, 999)}/2025"
    prev_ab, prev = pair(31, 58)
    dmft = pct(1.1, 2.9)
    docs.append(("caries_survey.pdf", [
        ["Dental caries among primary school learners in Buffalo City", "Abstract",
         f"Caries prevalence was {prev_ab}% and most lesions were untreated.",
         "Sampling is on page 2 and the full results on page 3."],
        ["2. Methods", f"Two calibrated dentists examined {n} learners aged 9 to 11 in {schools} schools.",
         "Caries was scored with the WHO DMFT index.",
         f"Ethics approval: {hrec}."],
        ["3. Results", f"Caries prevalence: {prev}%", f"Mean DMFT: {dmft}",
         "Table 5 breaks prevalence down by school quintile."]],
        [("caries_survey.pdf: how many learners were examined and what caries prevalence was found?",
          {"expect_all_norm": [n, prev]}),
         ("What ethics approval number does caries_survey.pdf cite, and what mean DMFT does it report?",
          {"expect_all_norm": [hrec, dmft]}),
         ("Compare the prevalence in caries_survey.pdf's abstract with its results page. Same number?",
          {"expect_all_norm": [prev_ab, prev]})]))

    # F: blue crane GPS tracking
    n, mb, longest = num(14, 38), num(300000000, 399999999), num(160, 420)
    d_ab, d = pair(8, 27)
    docs.append(("crane_tracking.pdf", [
        ["Movement of blue cranes across the Overberg wheatbelt", "Abstract",
         f"Tagged birds moved a median of {d_ab} km per day outside the breeding season.",
         "Tagging is described on page 2; movement results are on page 3."],
        ["2. Methods", f"We fitted {n} adult blue cranes with solar GPS tags.",
         "Tags logged a position every 15 minutes.",
         f"Data: Movebank study ID {mb}."],
        ["3. Results", f"Median daily distance: {d} km", f"Longest single flight: {longest} km",
         "Figure 4 shows roost sites by month."]],
        [("How many cranes were tagged in crane_tracking.pdf, and what was the longest single flight?",
          {"expect_all_norm": [n, longest]}),
         ("crane_tracking.pdf: what's the Movebank study ID, and which page gives it?",
          {"expect_norm": [mb]}),
         ("Is the median daily distance in crane_tracking.pdf's abstract consistent with the results section?",
          {"expect_all_norm": [d_ab, d]})]))

    # G: LFP battery ageing (4 pages, appendix)
    n, temp, rm = num(36, 72), num(25, 45), num(9, 14)
    dry = f"dryad.{''.join(r.choice('abcdefghjkmnpqrstuvwxyz0123456789') for _ in range(8))}"
    ret_ab, ret = pair(78, 93)
    docs.append(("battery_ageing.pdf", [
        ["Capacity fade of LFP cells under fast charging", "Abstract",
         f"Cells retained {ret_ab}% of their capacity after 2000 cycles.",
         "Test conditions: page 2. Results: page 3. Excluded cells: Appendix A, page 4."],
        ["2. Methods", f"We cycled {n} LFP pouch cells at {temp} C with 2C charging.",
         "Capacity was checked every 100 cycles at C/10.",
         f"Data availability: doi:10.5061/{dry}."],
        ["3. Results", f"Capacity retention after 2000 cycles: {ret}%",
         "Resistance rose fastest in the first 300 cycles.",
         "Figure 5 compares fade at two charge rates."],
        ["Appendix A", f"Cells removed from analysis: {rm} (thermal fault or tab damage).",
         "Their partial data are included in the archive."]],
        [("battery_ageing.pdf: at what temperature were the cells cycled, and what capacity retention was measured?",
          {"expect_all_norm": [temp, ret]}),
         ("Where is battery_ageing.pdf's data archived? Give the identifier.",
          {"expect_norm": [dry]}),
         ("How many cells does battery_ageing.pdf's appendix say were excluded, and out of how many cycled?",
          {"expect_all_norm": [rm, n]})]))

    # H: isiXhosa speech recognition
    hours, spk, rma = num(60, 190), num(140, 480), f"RMA-{num(1000, 9999)}"
    w1, w2 = pct(9, 15), pct(19, 33)
    wer_ab, wer = pair(13, 22)
    docs.append(("asr_xhosa.pdf", [
        ["Speech recognition for isiXhosa with self-supervised pretraining", "Abstract",
         f"Our best system reaches a word error rate of {wer_ab}%.",
         "The corpus is described on page 2 and the evaluation on page 3."],
        ["2. Data and methods", f"Training used {hours} hours of read speech from {spk} speakers.",
         f"Corpus: SADiLaR resource {rma}.",
         "Test speakers do not appear in training."],
        ["3. Evaluation", "Table 6. Word error rate by condition",
         f"Clean audio {w1}%   Noisy audio {w2}%", f"Overall WER: {wer}%",
         "Most errors are in loanwords and numbers."]],
        [("asr_xhosa.pdf: how many hours of speech were used for training, and what overall word error rate was reached?",
          {"expect_all_norm": [hours, wer]}),
         ("What WER does asr_xhosa.pdf report on noisy audio, and how many speakers were recorded?",
          {"expect_all_norm": [w2, spk]}),
         ("Which corpus identifier does asr_xhosa.pdf give for its training data, and on what page?",
          {"expect_norm": [rma]})]))

    # I: road crash hotspots
    n, ref, top_c = num(1100, 4900), f"RTMC-{num(10000, 99999)}", num(40, 95)
    road = r.choice(["N2 at Mthatha", "R61 at Komani", "N6 at Stutterheim"])
    sh_ab, sh = pair(34, 61)
    docs.append(("crash_hotspots.pdf", [
        ["Road crash hotspots in the Eastern Cape", "Abstract",
         f"{sh_ab}% of recorded crashes happened at intersections.",
         "Data sources are on page 2 and hotspot results on page 3."],
        ["2. Data", f"We analysed {n} crashes from police records between 2021 and 2024.",
         "Crash locations were snapped to the provincial road network.",
         f"Data request reference: {ref}."],
        ["3. Results", f"Top hotspot: {road} with {top_c} crashes",
         f"Share at intersections: {sh}%", "Night-time crashes were twice as likely to be fatal."]],
        [("crash_hotspots.pdf: which road is the top hotspot, and how many crashes were analysed in total?",
          {"expect_all_norm": [road, n]}),
         ("Does crash_hotspots.pdf's abstract give the same intersection share as its results?",
          {"expect_all_norm": [sh_ab, sh]}),
         ("What data request reference does crash_hotspots.pdf cite?",
          {"expect_norm": [ref]})]))
    return docs


# --------------------------------------------------------------------------- humanizer
# (file, prompt, numbers of which one must survive in the reply or the edit, text). Written for
# run 2; none shares a sentence, a file name or a number with the gate's three.

HUMANIZER_TEXTS = [
    ("notes/newsletter.md", "Humanize notes/newsletter.md.", ["214"],
     "In an ever-evolving landscape, our community garden stands as a beacon of hope. This "
     "season, our dedicated volunteers harvested 214 kilograms of vegetables, a testament to the "
     "power of collective action. It's not just about growing food - it's about growing "
     "connections. Together, we are cultivating a brighter future.\n"),
    ("cover_letter.txt", "cover_letter.txt sounds robotic. Rewrite it in a natural voice.", ["11", "eleven"],
     "I am writing to express my profound enthusiasm for the Data Analyst role at Meridian "
     "Logistics. With 11 years of experience navigating complex datasets, I bring a unique blend "
     "of analytical rigor and creative problem-solving. I am confident that my skills would "
     "seamlessly align with your team's vision, and I look forward to the opportunity to "
     "contribute to your continued success.\n"),
    ("listing.md", "Edit listing.md so it stops reading like generated marketing copy.", ["860"],
     "Welcome to your dream home! This stunning four-bedroom residence boasts breathtaking "
     "mountain views and an abundance of natural light. Situated on a generous 860 square metre "
     "stand, it offers the perfect blend of comfort, elegance, and functionality. Don't miss this "
     "once-in-a-lifetime opportunity to elevate your lifestyle.\n"),
    ("release_notes.md", "Remove the AI tells from release_notes.md.", ["58"],
     "We're excited to announce a game-changing update that takes your workflow to the next "
     "level! Sync is now 58 percent faster, empowering you to stay seamlessly connected across "
     "all your devices. But that's not all - we've also reimagined the settings page from the "
     "ground up. Your productivity journey starts here.\n"),
    ("linkedin.txt", "My linkedin.txt post is stuffed with AI cliches. Fix the wording.", ["18", "eighteen"],
     "I'm humbled and honored to share that after 18 months of relentless dedication, our team "
     "has officially launched our new logistics platform. This journey has taught me that "
     "success isn't about the destination - it's about the people you meet along the way. "
     "Grateful. Inspired. Ready for what's next.\n"),
    ("notes/school_update.md", "Rewrite notes/school_update.md in plain, human language.",
     ["9 400", "9400", "9,400"],
     "We are absolutely delighted to share some truly heartwarming news with our incredible "
     "school community! Through their boundless creativity and unwavering determination, our "
     "Grade 7 learners raised R 9 400 for the library. This remarkable achievement is a shining "
     "example of what can happen when passion meets purpose.\n"),
    ("bakery.txt", "bakery.txt has that ChatGPT gloss. Rewrite it so a real baker could have written it.", ["350"],
     "Step into a world of flavour at Crumb & Co, where every loaf tells a story. Our passionate "
     "artisans bake 350 loaves every single morning, crafting each one with love, care, and the "
     "finest ingredients. More than just a bakery, we are a celebration of tradition, community, "
     "and the simple joy of fresh bread.\n"),
    ("charity_appeal.md", "Take the AI-speak out of charity_appeal.md.", ["150"],
     "This winter, you have the power to make a profound difference. Every R 150 you give "
     "provides a warm blanket to a family in need, transforming a cold night into a moment of "
     "comfort and dignity. Together, we can create a ripple effect of kindness that touches "
     "countless lives. Your generosity isn't just a gift - it's a lifeline.\n"),
    ("gym_promo.txt", "Rework gym_promo.txt so it doesn't read like an AI wrote it.", ["5.4"],
     "Unleash your inner athlete at IronPeak Fitness! Our state-of-the-art facility and "
     "world-class trainers are here to help you crush your goals. Last year, members on our "
     "signature programme lost an average of 5.4 kg in ten weeks. It's more than a gym. It's a "
     "lifestyle. Your transformation starts today.\n"),
    ("notes/conference_recap.md", "Clean up notes/conference_recap.md - it's full of AI phrasing.", ["600", "22"],
     "The 2026 Data for Good Summit was nothing short of extraordinary. Bringing together over "
     "600 delegates from 22 countries, the event served as a vibrant hub of innovation, "
     "collaboration, and thought leadership. Attendees delved into cutting-edge topics, forged "
     "meaningful connections, and left inspired to drive lasting change.\n"),
    ("app_description.txt", "Humanise the wording in app_description.txt.", ["900"],
     "Introducing StudyNest - the ultimate all-in-one platform designed to revolutionize the way "
     "learners study. Trusted by more than 900 schools, StudyNest seamlessly integrates smart "
     "flashcards, personalized quizzes, and real-time progress tracking. Whether you're cramming "
     "for exams or building lifelong skills, StudyNest empowers you to unlock your full "
     "potential.\n"),
    ("restaurant_review.md", "restaurant_review.md reads as machine-written. Rewrite it like a person who ate there.", ["14", "fourteen"],
     "From the moment we stepped through the doors, we were transported on a culinary journey "
     "like no other. The lamb shank, slow-cooked for 14 hours, was a symphony of flavours that "
     "danced on the palate. The ambiance struck the perfect balance between elegance and warmth. "
     "Truly, this is a hidden gem that deserves every accolade.\n"),
    ("notes/memo.md", "Rewrite notes/memo.md without the corporate AI voice.", ["65"],
     "We are pleased to inform you that, as part of our ongoing commitment to fostering a "
     "dynamic and flexible work environment, the office will be transitioning to a hybrid model. "
     "To facilitate this exciting change, we will maintain 65 bookable desks. We believe this "
     "initiative will empower teams to thrive and achieve unprecedented synergy.\n"),
    ("travel_blog.md", "travel_blog.md sounds like a chatbot. Make it read like a real trip report.", ["27"],
     "Nestled among towering peaks, the Amphitheatre trail offers an experience that will leave "
     "you utterly breathless. Our 27 kilometre hike was a testament to the raw, untamed beauty of "
     "the Drakensberg. Every step revealed a new vista, each more awe-inspiring than the last. "
     "It wasn't just a hike - it was a journey of the soul.\n"),
    ("podcast_blurb.txt", "Strip the AI flavour out of podcast_blurb.txt.", ["73"],
     "In this captivating episode 73, we dive deep into the fascinating world of urban "
     "beekeeping. Join us as we explore the buzzing intersection of nature, community, and "
     "sustainability. You won't want to miss these game-changing insights that will transform "
     "the way you think about honey forever!\n"),
    ("notes/volunteer_thanks.md", "Rewrite notes/volunteer_thanks.md so it sounds sincere, not generated.",
     ["1 280", "1280", "1,280"],
     "To our amazing volunteers: words cannot express our heartfelt gratitude. Your 1 280 hours "
     "of selfless service this year have been nothing short of transformative. You are the "
     "backbone of our organization, the heartbeat of our mission, and the reason we do what we "
     "do. Thank you for being the change.\n"),
    ("tutoring.txt", "Fix tutoring.txt: it reads like AI copy.", ["19", "nineteen"],
     "At BrightPath Tutoring, we believe every learner deserves the chance to shine. Our "
     "dedicated team of expert educators delivers personalized, results-driven support that "
     "unlocks each child's unique potential. Last term, our learners improved their maths marks "
     "by an average of 19 percent. Because when learners thrive, futures flourish.\n"),
    ("event_invite.md", "Make event_invite.md sound like a person wrote it.", ["40", "forty"],
     "You are cordially invited to an unforgettable evening of connection, inspiration, and "
     "celebration! Join us on 28 November as we come together to honour the visionaries shaping "
     "our city's creative landscape. Seating is limited to 40 guests, so be sure to secure your "
     "spot for this truly magical night.\n"),
    ("bookshop.txt", "Please rewrite bookshop.txt without the AI-sounding filler.", ["2 600", "2600", "2,600"],
     "Discover a hidden treasure trove for book lovers in the heart of Makhanda! Our charming "
     "shop houses 2 600 carefully curated second-hand titles, each waiting to whisk you away on "
     "an unforgettable adventure. Whether you're a seasoned bibliophile or a curious newcomer, "
     "there's something magical for everyone.\n"),
    ("notes/product_page.md", "notes/product_page.md is generic AI copy. Rewrite it plainly.", ["1.7"],
     "Elevate your kitchen experience with the AuraBoil Pro, where sleek design meets "
     "cutting-edge performance. This revolutionary kettle boils 1.7 litres in record time, "
     "seamlessly blending style and functionality. It's not just a kettle - it's a statement. "
     "Experience the future of tea time today.\n"),
    ("municipal_notice.txt", "De-fluff municipal_notice.txt so it reads like a normal notice from a person.", ["36"],
     "Dear valued residents, we wish to inform you that, in our ongoing commitment to delivering "
     "world-class service, essential maintenance will be undertaken on the main water line. "
     "Supply will be interrupted for 36 hours. We sincerely appreciate your patience and "
     "understanding as we work tirelessly to build a better tomorrow together.\n"),
    ("notes/team_update.md", "Rewrite notes/team_update.md the way an engineer would actually write it.", ["128"],
     "What an incredible sprint it has been! Our rockstar team closed 128 tickets, demonstrating "
     "unparalleled dedication and seamless collaboration. This milestone isn't just a number - "
     "it's a reflection of our relentless pursuit of excellence. Let's keep this momentum going "
     "and continue to raise the bar!\n"),
    ("club_news.txt", "club_news.txt sounds auto-generated. Can you make it sound like our club secretary?", ["15", "fifteen"],
     "Exciting times are ahead for the Riverside Running Club! This month, we proudly welcomed "
     "15 new members into our vibrant and ever-growing family. Each brings a unique energy that "
     "enriches our community and fuels our shared passion for running. Lace up, step out, and "
     "let's make every kilometre count!\n"),
    ("notes/about_us.md", "Humanize notes/about_us.md; keep the facts.", ["31", "thirty-one"],
     "Founded on a passion for innovation, Kloofwood Joinery has become a trusted name in bespoke "
     "furniture. Our talented team of 31 craftspeople brings decades of combined expertise to "
     "every project, delivering timeless pieces that seamlessly blend form and function. At "
     "Kloofwood, we don't just build furniture - we build legacies.\n"),
    ("course_blurb.txt", "Rewrite course_blurb.txt so it doesn't sound like an AI sales pitch.", ["10", "ten"],
     "Embark on a transformative learning journey with our comprehensive Introduction to Data "
     "Analysis course. Over 10 immersive weeks, you will master essential tools, unlock powerful "
     "insights, and gain the confidence to thrive in today's data-driven world. Don't just "
     "learn - evolve. Your future self will thank you.\n"),
    ("notes/grant_summary.md", "notes/grant_summary.md was drafted by AI and it shows. Rewrite it.", ["450"],
     "This groundbreaking initiative seeks to revolutionize access to clean energy in "
     "underserved communities. By harnessing the transformative power of solar technology, the "
     "project will reach 450 households, fostering resilience, empowerment, and sustainable "
     "growth. Ultimately, it represents a bold step towards a brighter, more equitable future "
     "for all.\n"),
]


# --------------------------------------------------------------------------- coffee
# Answered from report.py over the tracker's real SQLite. Different wording, windows and
# questions from the gate's four (most specials by retailer / 90 days, quick summary / 45 days,
# PnP-vs-Takealot usually cheaper, lowest 95g price) and from run 1's ten.

COFFEE_RUN2 = [
    "Using the coffee tracker's data, what's the highest price recorded for the 200g Kronung refill over the last 120 days?",
    "Which Kronung product had the biggest price swing at Checkers over the past 100 days, per the tracker report?",
    "Run the coffee tracker report for the last 7 days and tell me what stands out.",
    "What did the tracker see for the Kronung sachets at Pick n Pay over the past 3 weeks?",
    "How does Checkers' Kronung pricing compare with Pick n Pay's over the last 180 days, per the tracker?",
    "Per the price tracker, what's the cheapest Kronung listing on Takealot in the past 75 days?",
    "Over the last 110 days, which retailer's Kronung prices have been the steadiest, according to the coffee tracker?",
    "From the coffee tracker report, what's the mean price of the 47.5g Kronung jar at each retailer?",
    "Has the Takealot case of Kronung changed price lately? Check the tracker history.",
    "How many price observations has the coffee tracker logged for Takealot in the last 50 days?",
    "Pull the coffee tracker report for the past 10 days and give me the headline numbers.",
    "According to the coffee price history, which Checkers Kronung listings swing by more than 10% (CoV)?",
    "What's the price range for Kronung freeze-dried at Pick n Pay over the last 40 days, per the tracker?",
    "Look at the coffee tracker's last 6 months: is there any retailer where Kronung prices move a lot?",
    "Checkers or Takealot: which averaged lower for the Kronung 200g jar over the last 21 days, going by the tracker?",
    "What does the coffee tracker say about Kronung stick packs on Takealot over the last 80 days?",
    "Kronung price report from the tracker for the past 120 days, broken down by retailer please.",
    "Which single Kronung listing has the most observations in the tracker over the last 35 days?",
    "In the tracker's data for the last 14 days, did any Kronung product show specials at Takealot?",
    "Coffee tracker: what's the most expensive Kronung item recorded in the past 150 days, and where?",
    "Using the tracker's report, which Pick n Pay Kronung product had the most specials over the last 70 days?",
    "Give me the coffee tracker's view of Kronung prices at Checkers for the last 25 days.",
    "What's the average price the tracker has for the Kronung 300g on Takealot over the last 365 days?",
    "Check the coffee tracker history: what min and max did it record for the Kronung sachet box of 26 at Pick n Pay in the last 100 days?",
    "Per the tracker report for the last 55 days, how many distinct Kronung products does each retailer list?",
]


# --------------------------------------------------------------------------- items

def _gate():
    import skill_gate_wide as W
    return W, W.items_all()


def _gate_values(gate_items: list[dict]) -> set[str]:
    vals: set[str] = set()
    for it in gate_items:
        s = it["spec"]
        for k in ("expect_norm", "expect_all_norm", "expect_num", "expect"):
            vals.update(str(v) for v in s.get(k) or [])
    return {v for v in vals if len(v) >= 2}


def items() -> list[dict]:
    """Run-2 training items in build_skill_anchor's shape: {id, family, prompt, spec, fixture}.
    spec["judge"] = "wide" makes the builder judge with skill_gate_wide.judge."""
    _, gate = _gate()
    banned = _gate_values(gate)
    out: list[dict] = []
    for d, (fn, pages, qs) in enumerate(_pdf_docs(banned)):
        for q, (prompt, extra) in enumerate(qs):
            out.append({"id": f"pdfx-{d}-{q}", "family": "pdf-explore", "prompt": prompt,
                        "fixture": {"pdfs": {fn: pages}},
                        "spec": {"skill": PDF_SKILLS, "script_re": PDF_SCRIPT_RE, "judge": "wide",
                                 **extra}})
    for i, (fn, prompt, nums, text) in enumerate(HUMANIZER_TEXTS):
        out.append({"id": f"hum-{i:02d}", "family": "humanizer", "prompt": prompt,
                    "fixture": {"files": {fn: text}},
                    "spec": {"skill": HUMANIZER_SKILLS, "must_read_re": re.escape(fn.split("/")[-1]),
                             "expect_num": nums, "strict_fences": False, "judge": "wide"}})
    for i, prompt in enumerate(COFFEE_RUN2):
        out.append({"id": f"coffee2-{i:02d}", "family": "coffee", "prompt": prompt,
                    "spec": {"skill": "coffee-price-tracker", "script_re": r"report\.py",
                             "ground_alt": {"names": RETAILERS}, "judge": "wide"}})
    return out


def _fixture_texts(it: dict) -> list[str]:
    fx = it.get("fixture") or {}
    t = list((fx.get("files") or {}).values())
    t += ["\n".join(ln for page in pages for ln in page) for pages in (fx.get("pdfs") or {}).values()]
    if "text" in fx:
        t.append(fx["text"])
    if "lines" in fx:
        t.append("\n".join(fx["lines"]))
    return t


def _fixture_names(it: dict) -> set[str]:
    fx = it.get("fixture") or {}
    names = set((fx.get("files") or {})) | set((fx.get("pdfs") or {}))
    names |= {fx[k] for k in ("file", "pdf") if k in fx}
    return {Path(n).name for n in names}


def _shingles(s: str, n: int = 8) -> set[str]:
    w = re.findall(r"[a-z0-9']+", s.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _has_token(text: str, v: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(v)}(?![\w]|\.\d)", text, re.I) is not None


def check_disjoint(run2: list[dict], run1: list[dict]) -> None:
    """Refuse to build if a run-2 prompt or fixture leaks into the held-out gates (see module
    docstring for the rules)."""
    from skill_gate import gate_items
    _, gate = _gate()
    problems: list[str] = []
    seen = {it["prompt"].strip() for it in gate} | {it["prompt"].strip() for it in gate_items()} \
        | {it["prompt"].strip() for it in run1}
    for it in run2:
        if it["prompt"].strip() in seen:
            problems.append(f"{it['id']}: prompt equals a gate/run-1 prompt")
    if len({it["prompt"] for it in run2}) != len(run2):
        problems.append("duplicate run-2 prompts")
    gate_names = set().union(*(_fixture_names(g) for g in gate))
    gate_texts = [t for g in gate for t in _fixture_texts(g)]
    gate_sh = set().union(*(_shingles(t) for t in gate_texts))
    gate_vals = _gate_values(gate)
    for it in run2:
        if _fixture_names(it) & gate_names:
            problems.append(f"{it['id']}: fixture name {_fixture_names(it) & gate_names} is the gate's")
        for t in _fixture_texts(it):
            if _shingles(t) & gate_sh:
                problems.append(f"{it['id']}: fixture shares text with a gate fixture")
            for v in gate_vals:
                if _has_token(t, v):
                    problems.append(f"{it['id']}: fixture contains gate answer value {v!r}")
        s = it["spec"]
        for k in ("expect_norm", "expect_all_norm", "expect_num"):
            for v in s.get(k) or []:
                # Same-family gate fixtures only: small numbers ("14", "40") occur all over
                # the other families' fixtures (dates, counts) and teach nothing about them.
                fam_texts = [t for g in gate if g["family"] == it["family"] for t in _fixture_texts(g)]
                if len(str(v)) >= 2 and not re.fullmatch(r"[A-Za-z ]+", str(v)) and \
                        any(_has_token(gt, str(v)) for gt in fam_texts):
                    problems.append(f"{it['id']}: answer value {v!r} appears in a gate fixture")
    if problems:
        raise SystemExit("run-2 items are not disjoint from the gate:\n  " + "\n  ".join(sorted(set(problems))))
