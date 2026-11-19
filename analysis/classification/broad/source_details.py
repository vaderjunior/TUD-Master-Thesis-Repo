from pathlib import Path
import json
import numpy as np
from analysis.classification.broad import sources as s

NOTES = {
    "S01": "Unanimous offensive gold is mapped to non-hate. Full attends to praise of makeup skills, while small treats the slur itself as sufficient. The saved text does not establish in-group reclamation; the full rationale presents that only as a possibility.",
    "S02": "Unanimous offensive/non-hate gold. Full recognizes that the speaker rejects the reported claim but nevertheless attributes hateful endorsement or normalization to the report; small and zero distinguish quotation/criticism from endorsement. Fine and legal labels have no gold here.",
    "S03": "The sole stable full/small FN fix in this source. Two annotators called this hate and one normal. Full interprets animal language as a racial/welfare allusion; the text is context-poor, so agreement with majority gold is not proof that its inferred background is established.",
    "S04": "Two hate votes and one offensive vote. Full treats the passage as political criticism of political Islam; small treats it as a religious generalization. This is an annotation/definition boundary, not evidence that new labels should be substituted.",
    "S05": "Full treats the bare ideology phrase as a mention, small and zero as endorsement. The full rationale mentions provided examples, but that is an output claim; actual retrieved context and causal influence remain Part 6.",
    "S06": "Full expands a violent statement about ISIS into an attack on a religious group; small and zero keep the target as a political/terrorist organization. Non-hate gold here is a narrow hate-speech classification, not an endorsement of violence.",
    "S07": "Full and zero identify an implicit ideological signal where small requires more explicit derogation. Full changes fine labels across repeats while keeping the binary decision; the binary fix does not prove a fully correct explanation.",
    "S08": "Full accepts a book-title reading of the short slogan and misses the original implicit-hate label. Small identifies an ideological signal. The publication/background claims in both rationales are not independently established by this analysis.",
    "S09": "The non-hate MHS score is -1.12. Full reads a violent narrative without identity-directed endorsement; small and zero treat its racial wording as hate. Violent subject matter and hate gold are different constructs.",
    "S10": "The MHS score 0.25 remains below the frozen hate boundary. Full converts a trauma account and self-directed distress into claims of incitement/threat against men; small and zero distinguish the outburst from a group attack. The record is a dataset example, not a current disclosure.",
    "S11": "MHS intensity score 3.05 is hate/high gold. Full recognizes the gendered abuse and repairs the gate, but its saved first answer has low severity while its rationale says medium. Binary agreement is not complete-answer or rationale consistency.",
    "S12": "MHS score 2.64 from one annotator is hate gold. Full/zero classify a personal insult as non-hate while small infers a gender generalization. The single-annotator intensity record and sparse conversational context limit interpretation; gold stays unchanged.",
    "S13": "DeTox hate score zero. Full interprets the reptilian reference as fictional/conspiracy mockery; small treats it as group dehumanization. Neither this binary endpoint nor this review validates the emitted legal conclusions.",
    "S14": "DeTox hate score zero. Full adds ethnic/religious background and derogatory intent not established by the short query. Small makes a similar background assertion but returns non-hate. Correct binary output therefore does not certify a sound rationale.",
    "S15": "DeTox hate score .667. Full calls the political/occupational collective other and agrees with source hate gold; small excludes it for lack of a protected trait. This contrasts with HateCheck unprotected-group cases: broad toxicity and protected-group definitions do not coincide.",
    "S16": "Unanimous DeTox hate score. Full emphasizes personal encouragement and overlooks the nationality-directed closing insult; small and zero return hate. Full legal output varies despite a stable negative hate gate, consistent with the independent legal branch.",
    "S17": "GAHD R3 non-hate. Full reads a descriptive gender observation, small treats the generalization as stereotyping. Full also speculates that the attributed name is a typo; that extra story is unsupported by the input and does not establish correctness of the reasoning.",
    "S18": "GAHD R2 non-hate personal abuse. Full assigns gender hate to accusations about an individual and family; small and zero keep individual misconduct distinct from group-directed hate. Offensiveness alone is not the gold criterion.",
    "S19": "GAHD R2 hate gold. Full agrees with gold but introduces infertility/reproductive capacity, absent from the query. This is a correct gate with an unsupported query-level explanation; Part 6 must inspect the actual context before attributing the added story to retrieval.",
    "S20": "GAHD R2 hate gold. Full and zero frame the criticism of girls and young women as social concern; small identifies a broad gender stereotype. This is another conservative-boundary miss rather than a failed parser.",
    "S21": "All nine decisions miss the group-directed negative-emotion test. The rationales require extra severity or explicit derogatory markers even though the HateCheck function tests hostility itself. This is a persistent failure across retrieval settings.",
    "S22": "Full acknowledges a negative stereotype about women leaders but dismisses it as political opinion and returns non-hate in all repeats. Small/zero detect it. Full thus can recognize the content yet apply a stricter classification boundary.",
    "S23": "No valid full-setting error exists in this function under the frozen selection rule; retain the empty slot. Perfect saved performance does not establish a zero future error rate.",
    "S24": "Full dismisses lack of concern for immigrants lives as insufficiently explicit. Small/zero detect implicit devaluation. This is an example of the missed-implicit-hate cost accompanying fewer false alarms.",
    "S25": "No valid full-setting error exists in direct threats under the rule; retain the empty slot and the finite diagnostic sample limitation.",
    "S26": "Full treats women like you as merely personal and misses an exclusionary normative-threat test, while small/zero return hate. A legal flag under the negative gate does not repair the hate error.",
    "S27": "Full overextends an ambiguous/reclaimed reading to a dismissive use of queer; small/zero identify the directed slur. Improvements on reclaimed slurs coexist with failures on hateful uses.",
    "S28": "A smoking context disambiguates a slur homonym. Full is correct in its first repeat but wrong in the other two, while small/zero fail all three. The case is deliberately the maximum-error example, not evidence that full never recognizes homonyms.",
    "S29": "All settings misclassify the supportive shout-out. Full improves the reclaimed-slur function overall but still treats this occurrence as inherently hateful. Do not infer speaker identity or in-group membership beyond the benchmark annotation.",
    "S30": "Full reads women like you as a personal grievance and misses the hateful-profanity function in all repeats. Small detects it; zero varies. This reflects the same group-versus-individual boundary seen in the natural sources.",
    "S31": "No valid full-setting error in non-hateful profanity; retain the empty slot.",
    "S32": "No valid full-setting error in reference across a later clause; retain the empty slot.",
    "S33": "No valid full-setting error in reference across a later sentence; retain the empty slot.",
    "S34": "Full treats any woman like you as individualized disdain and misses the negated-positive test. Small returns hate, zero misses it. The raw reference link is retained; this one answer is not a standalone causal negation effect.",
    "S35": "Full invents possible sarcasm or a prior hostile context to override an explicit disavowal of hatred. Small and zero vary. The text does not supply the prior history suggested by the rationale.",
    "S36": "Full dismisses group-directed disgust in question form as insufficiently intense. Small/zero detect hate. The same child appears again in linked case S51 and remains a duplicate, not a new independent example.",
    "S37": "No valid full-setting error in hate phrased as an opinion; retain the empty slot.",
    "S38": "No valid full-setting error in neutral identity mentions; retain the empty slot.",
    "S39": "All settings invent a sarcastic reading of a positive disability statement. This persistent positive-identity false alarm qualifies the strong aggregate performance on that function.",
    "S40": "The statement condemns a quoted wish of death. Full and small explicitly notice the condemnation but still classify the quoted content as the speakers hate; zero also fails. This is failure to use stance, not merely failure to see quotation marks.",
    "S41": "Full attributes the quoted/referenced insult to the speaker despite the explicit challenge to hateful speech. Small/zero correctly distinguish condemnation from endorsement. The context inspection remains separate.",
    "S42": "No valid full-setting error in abuse of objects; retain the empty slot.",
    "S43": "A violent threat against an unspecified individual is non-hate under this benchmark definition. Full changes across repeats and small consistently infers gender targeting from a pronoun. Non-hate does not mean safe, acceptable, or legally innocuous.",
    "S44": "All settings generalize other to an occupational/political collective and fail the unprotected-group boundary. This has the opposite source-label consequence from the DeTox Querdenker case S15, illustrating construct mismatch.",
    "S45": "Full fails one repeat on swapped spelling while small/zero detect hate. Its rationale understands the statement but rejects it for lack of explicit hostility, so this case does not establish an inability to decode the misspelling.",
    "S46": "Full interprets the nonstandard spelling as benign despite the surrounding categorical insult. Small/zero detect it. The rationale introduces a possible spelling context rather than responding to the full sentence.",
    "S47": "Full decodes the spaced slur but dismisses its protected-gender connection, whereas small/zero detect hate. The saved failure cannot simply be described as failure to read obfuscated characters.",
    "S48": "Full understands the merged-word disgust statement but requires violence/exclusion/inferiority to classify it as hate. Small/zero detect it. Recognition of words and the decision boundary are distinct.",
    "S49": "Full reads the leetspeak slur as sexual/moral personal abuse rather than protected-gender abuse; small/zero vary. The rationale decodes the term, so decoding failure is not established.",
    "S50": "Same-gold linked pair: full gets both the question and direct statement right; small gets the question right but misses the parent. This is a stable both-correct improvement, not necessarily an effect on the child alone.",
    "S51": "Same-gold linked pair: full misses the question every time and varies on the parent; small gets both right. In two full repeats the two answers are consistently negative and both wrong, directly illustrating why relation consistency alone is insufficient.",
    "S52": "Changed-gold linked pair: all settings detect the original threat, but only full consistently distinguishes its condemnation in the child. This is a clear within-benchmark both-correct gain.",
    "S53": "Changed-gold linked pair: all settings detect the original devaluation, but full wrongly attributes the condemning quote to its speaker; small gets both right. This retains a counterexample to the average changed-gold gain.",
}


def run():
    cases = s.c.read(s.BASE / "cases.json")
    s.require(set(NOTES) == {v["case_id"] for v in cases}, "Missing manual case note")
    s.r.verify(s.c.read(s.BASE / "focused_source_bindings.json"))
    s.save(
        s.BASE / "manual_review.json",
        dict(
            status="all_frozen_cases_and_fixed_rationales_read",
            case_slots=len(cases),
            empty_slots=[v["case_id"] for v in cases if not v["key"]],
            notes=NOTES,
            scope="All nine delivered decisions per selected input and first-repeat full/small rationales; 53 deterministic slots, with original empty slots and duplicates retained. Qualitative observations are illustrations, not prevalence or causal estimates. No relabeling.",
            sources=s.bindings(
                [
                    Path(__file__),
                    s.BASE / "cases.json",
                    s.BASE / "case_selection.json",
                    s.BASE / "focused_source_bindings.json",
                ]
            ),
        ),
    )
    natural = s.c.read(s.BASE / "natural_binary.json")
    functions = s.c.read(s.BASE / "hatecheck_functions.json")
    targets = s.c.read(s.BASE / "hatecheck_target_functions.json")
    ident = s.c.read(s.BASE / "hatecheck_identities.json")
    fine = s.c.read(s.BASE / "fine_sources.json")
    out = {}
    changes = {}
    for key, g in natural.items():
        changes[key] = {
            p["reference"]: {
                k: p["full"]["mean"][k] - p["reference_summary"]["mean"][k]
                for k in ("fp", "fn", "n")
            }
            for p in g["pairs"]
        }
    out["natural_paired_count_changes"] = changes
    out["English_full_small_FP_savings_share"] = {
        key: -v["rag_small"]["fp"]
        / sum(
            -x["rag_small"]["fp"] for k, x in changes.items() if k.startswith("en_test")
        )
        for key, v in changes.items()
        if key.startswith("en_test")
    }
    out["fine_source_absent_fixed_macro_labels"] = {
        key: [
            label
            for label in g["macro_labels"]
            if g["shared_gold_support"][g["labels"].index(label)] == 0
        ]
        for key, g in fine.items()
    }
    classification = {}
    for title, groups, metric in [
        ("binary_function", functions, "error_rate"),
        ("target_function", targets, "gate_aware_exact"),
    ]:
        classification[title] = {}
        for ref in ("rag_small", "zero_shot"):
            rows = []
            for fn, g in groups.items():
                p = next(p for p in g["pairs"] if p["reference"] == ref)
                ci = p["intervals"]["component"]["metrics"][metric]["ci95"]
                ii = p["intervals"]["item"]["metrics"][metric]["ci95"]
                tag = lambda a: (
                    "above_zero"
                    if a[0] > 0
                    else "below_zero" if a[1] < 0 else "includes_zero"
                )
                rows.append(
                    dict(
                        function=fn,
                        component=tag(ci),
                        item=tag(ii),
                        disagree=tag(ci) != tag(ii),
                    )
                )
            classification[title][ref] = rows
    out["resampling_sign_classification"] = classification
    out["women_share_of_HateCheck_full_false_negatives"] = ident["women/True"][
        "common_nine"
    ]["rag_full"]["mean"]["fn"] / sum(
        g["common_nine"]["rag_full"]["mean"]["fn"]
        for k, g in ident.items()
        if k.endswith("/True")
    )
    metadata = s.c.read(s.BASE / "endpoint_metadata.json")[
        "en_hatecheck_eval/target_group"
    ]
    with np.load(s.ROOT / metadata["path"], allow_pickle=False) as z:
        y = z["gold"]
        p = z["predicted"]
        v = z["valid"]
        gate = z["gate"]
    ix = ident["trans people/target_exact"]["common_indices"]
    tg = {}
    for ci, co in enumerate(s.CONDITIONS):
        rows = []
        for j in range(ci * 3, ci * 3 + 3):
            extra = p[j, ix].astype(bool) & ~y[ix].astype(bool)
            missing = ~p[j, ix].astype(bool) & y[ix].astype(bool)
            correct = (p[j, ix] == y[ix]).all(1) & gate[j, ix]
            sexual = metadata["labels"].index("sexual_orientation")
            rows.append(
                dict(
                    n=len(ix),
                    wrong=int((~correct).sum()),
                    gate_closed=int((~gate[j, ix]).sum()),
                    extra_sexual_orientation=int(extra[:, sexual].sum()),
                    extra_only=int((extra.any(1) & ~missing.any(1)).sum()),
                    missing_only=int((~extra.any(1) & missing.any(1)).sum()),
                    mixed=int((extra.any(1) & missing.any(1)).sum()),
                )
            )
        tg[co] = dict(
            repeats=rows,
            mean={k: float(np.mean([x[k] for x in rows])) for k in rows[0]},
        )
    out["trans_target_error_decomposition"] = tg
    s.save(
        s.BASE / "derived_findings.json",
        dict(
            status="descriptive_explanatory_decompositions_no_new_primary_inference",
            findings=out,
            sources=s.bindings(
                [Path(__file__)]
                + [
                    s.BASE / (f + ".json")
                    for f in (
                        "natural_binary",
                        "hatecheck_functions",
                        "hatecheck_target_functions",
                        "hatecheck_identities",
                        "fine_sources",
                        "endpoint_metadata",
                    )
                ]
                + [s.ROOT / metadata["path"]]
            ),
        ),
    )
    print(
        "Manual review recorded for all 53 slots; source-bound descriptive decompositions saved."
    )


if __name__ == "__main__":
    with s.r.original.offline():
        run()
