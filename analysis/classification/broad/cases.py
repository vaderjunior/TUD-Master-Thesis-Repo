from pathlib import Path
from analysis.classification.broad import errors as e

NOTES = {
    "C01": (
        "Sexual explicitness versus protected-group hate",
        "Full retrieval separates explicit sexual content from a protected-group attack. Small retrieval instead supplies gender, stereotyping/inferiority and low severity in every repeat.",
        "The fix is agreement with the preserved non-hate label. Sexual content alone does not establish a hateful target; no author identity or missing context is inferred.",
    ),
    "C02": (
        "Political insult mapped to a protected target",
        "Full retrieval classifies the political insults as hate and uses the other target, while small retrieval distinguishes political hostility from protected-group hate. All full repeats attach a legal label as well.",
        "The phrase about getting rid of people can be concerning independently of hate classification. The error is relative to this non-hate label, not a conclusion that the language is harmless or legally permitted.",
    ),
    "C03": (
        "Slur presence and an unresolved annotation boundary",
        "Both compared systems consistently classify the short slur-containing fragment as hateful toward sexual orientation. The frozen dataset gate is non-hate.",
        "There is too little visible context to confidently explain why the dataset label differs. Preserve the scored false positive without declaring the slur benign or silently relabelling the input.",
    ),
    "C04": (
        "Gendered personal abuse at the hate boundary",
        "Full retrieval treats the gendered sexual insult as hate and attaches gender/type/severity fields. Small retrieval calls it a personal insult and withholds those fields while retaining legal output.",
        "The gate fix matches the dataset, but disagreement over an individual insult versus a protected-characteristic attack is substantive. Legal presence is a separate output, not proof of criminality.",
    ),
    "C05": (
        "Literal reading versus an implicit racial interpretation",
        "Full retrieval reads ghetto blaster as a stereo term and the disaster reference as non-hateful humour. Small retrieval reads the same juxtaposition as racial mockery and matches the hateful gold gate.",
        "This is a stable new miss consistent with a lost implicit reading. The saved rationale provides the proposed association; it is not evidence about the actual retrieval mechanism or the speaker’s intent.",
    ),
    "C06": (
        "An implicit joke read literally",
        "Both systems interpret the black-computer joke literally and withhold all fine labels. The dataset marks it hateful, consistent with a possible racial-stereotype reading of the double meaning.",
        "This is a persistent benchmark miss, not evidence that either system cannot process the word black. The association remains an interpretation of this example, not a measured source-wide mechanism.",
    ),
    "C07": (
        "Informal slur use versus automatic hate assignment",
        "Full retrieval treats the phrase as informal conversational language, while no retrieval assigns race, hostile types and severity in all repeats. The non-hate gold gate favours full retrieval.",
        "A possible colloquial or reclaimed use can explain the disagreement, but the author’s identity and intent are not known. Do not infer group membership from language.",
    ),
    "C08": (
        "Organization reference broadened into a religious target",
        "Full retrieval maps the reference to ISIS into a religion target and returns high severity; no retrieval does not classify the quoted statement as protected-group hate.",
        "This is a scored regression. The violent wording remains concerning, but an organization reference is not by itself evidence that the speaker targeted everyone sharing a religion.",
    ),
    "C09": (
        "Slang and competing target assumptions",
        "Both systems treat the comparison of social labels across countries as hate. Their rationales give different accounts of what one slang label denotes, while targets vary across national origin, religion and other.",
        "The lexical explanations are unverified model claims. This persistent false positive also illustrates why a fluent rationale should not be accepted as an authoritative definition.",
    ),
    "C10": (
        "A scored fix with an unsupported gender assumption",
        "Full retrieval matches the hateful gold gate but assigns gender and interprets the insult as being about a woman. No retrieval calls it an individual insult; the visible text does not identify the person’s gender.",
        "The gate improvement does not demonstrate a sound explanation. This is a particularly useful caution against treating a correct binary answer as verified target attribution.",
    ),
    "C11": (
        "Sexual degradation versus the protected-target requirement",
        "Full retrieval calls the abusive sexual sentence non-hate because it lacks an explicit protected target. No retrieval matches the hateful gold gate, assigning other and multiple hostile types.",
        "This is a new miss under the frozen annotation. The visible text still leaves a definition boundary; no relabelling or claim about legal status follows.",
    ),
    "C12": (
        "Apparently inclusive wording with hateful gold",
        "Both systems read the statement about respect and defending whoever is attacked as non-hateful. That conflicts with the supplied implicit-hate gold gate.",
        "The supplied excerpt does not settle the disagreement. Missing context or annotation-boundary differences are possible; the case remains scored and is not removed to improve results.",
    ),
    "C13": (
        "Criticism of terminology mistaken for an identity attack",
        "Full retrieval reads criticism of terms such as real woman as discussion of language, while small retrieval assigns gender-related hate and sometimes sexual orientation.",
        "The fix agrees with non-hate gold. The visible speaker stance supports a distinction between discussing exclusionary language and endorsing it; no retrieved example was inspected here.",
    ),
    "C14": (
        "Personal criticism expanded into gender hate",
        "Full retrieval assigns gender hate and medium/high severity to a long personal attack about behaviour and parenting. Small retrieval calls it offensive personal criticism instead.",
        "The dataset marks it non-hate. That does not make the insults acceptable; the question is whether the protected characteristic is the basis of the attack rather than merely mentioned.",
    ),
    "C15": (
        "Dehumanizing wording and inferred group membership",
        "Both systems classify the statement about poachers as hateful. Small retrieval sometimes infers migrants, national origin or race even though those identities are not explicit in the text.",
        "The preserved gold is non-hate. The other-target definition and occupational/group boundary need care; the strongest observable concern is unsupported migration/race attribution, not a universal claim that this language cannot be hateful.",
    ),
    "C16": (
        "Gold agreement does not validate the explanation",
        "Full retrieval matches the hateful gold gate by describing refugees as targets of stereotyping/inferiority because the text says they suffer under the state. Small retrieval reads concern about their treatment.",
        "The text can plausibly be read as sympathetic to refugees. This scored fix therefore supplies no assurance of better reasoning; the label/text tension is retained unresolved.",
    ),
    "C17": (
        "Implicit nationality stereotype treated as an anecdote",
        "Full retrieval acknowledges a possible stereotype in the theft-in-Romania anecdote but calls it insufficient for hate. Small retrieval treats the unsurprised closing phrase as a nationality stereotype and matches gold.",
        "This stable regression illustrates a threshold difference for implicit derogation. It does not establish a causal effect of any particular retrieved context.",
    ),
    "C18": (
        "Political-personal insult versus the corpus label",
        "Both systems treat the attack on political figures as non-hate, conflicting with the German gold gate. Full retrieval nevertheless retains an insult/defamation output.",
        "The protected-target interpretation and the dataset boundary may differ. Model legal explanations are quoted evidence only and are not independently validated legal advice.",
    ),
    "C19": (
        "Reference and integration assumptions",
        "Full retrieval does not classify the migration/integration comment as hate. No retrieval reads discriminatory exclusion and supplies national-origin labels and medium/high severity.",
        "Pronouns and irony require context; the text mentions the speaker’s family’s migration background, which should not automatically be assigned to the other person. Gold favours full retrieval, without resolving every contextual assumption.",
    ),
    "C20": (
        "Counter-stereotype converted into hate",
        "Full retrieval marks the explicit rejection of a negative racial stereotype as low-severity hate. No retrieval preserves the non-hate gold label.",
        "This is a clear stance-handling regression in the saved wording: mentioning and rejecting a stereotype is not the same as endorsing it.",
    ),
    "C21": (
        "Implied superiority and ambiguous self-reference",
        "Both systems infer that the self-description of being white and heterosexual devalues other people. The dataset marks the input non-hate.",
        "The implication may be contested and context or irony is absent. Retain the false-positive classification against gold without declaring the statement unambiguously harmless.",
    ),
    "C22": (
        "An identity-related statement with a contested boundary",
        "Full retrieval interprets the two-genders statement as exclusion of other gender identities and matches the hateful gold gate. No retrieval calls it a non-hateful general assertion.",
        "The experiment measures agreement with this annotation and definition, not a universal ruling on every use of the statement or the speaker’s intent.",
    ),
    "C23": (
        "Religious-text criticism versus attacks on believers",
        "Full retrieval treats the dismissive religious statement as criticism of a text or belief; no retrieval treats it as religion-targeted hate and matches the gold gate.",
        "This is a scored miss at a substantive distinction between doctrine criticism and protected-group attack. The visible excerpt does not settle the corpus’s broader context.",
    ),
    "C24": (
        "Institutional/political abuse and the gold boundary",
        "Both systems call the criticism of police and a regime political speech rather than hate, while the dataset labels it hateful. Fine hate fields are consequently empty.",
        "This remains a benchmark miss. The result should not be used to redefine the dataset or make legal claims; occupation and institutional-target interpretations need later source-specific discussion.",
    ),
    "C25": (
        "A challenge to quoted hate correctly distinguished",
        "Full retrieval recognizes the question as challenging a quoted racist statement. Small retrieval attributes the quoted disgust to the input speaker and attaches race/type/severity outputs.",
        "This stable fix provides a concrete example of improved speaker-stance handling, but not its prevalence across all quotation cases.",
    ),
    "C26": (
        "A victim’s response mistaken for endorsement",
        "Full retrieval treats the statement reporting an insult against the speaker as denigration by the speaker. Small retrieval recognizes a response to prior hostility and matches non-hate gold.",
        "The direction of attribution is visible in the text. This regression counterbalances the successful quotation example; a mechanism is not uniformly solved.",
    ),
    "C27": (
        "Condemnation of a generalization ignored",
        "Both systems focus on the quoted sexist generalization and overlook that the sentence criticizes the person writing it. They consistently output gender hate.",
        "The persistent false positive demonstrates that a fluent explanation of the embedded insult can still misidentify the stance of the complete input.",
    ),
    "C28": (
        "Individual threat and protected identity boundary",
        "Full retrieval matches the hateful diagnostic gate for a threat against a woman and supplies gender/threat/severity fields. Small retrieval withholds hate because it sees an individual rather than a group.",
        "A protected-characteristic attack can target an individual; however, mere identity mention is not a universal proof of motive. This is a result on the diagnostic’s preserved label.",
    ),
    "C29": (
        "Devaluation without an explicit threat",
        "Full retrieval treats contempt for concern about immigrants’ lives as insufficiently explicit hate. Small retrieval identifies devaluation of the group and matches gold.",
        "The missed target output follows the negative hate gate. This case illustrates a stricter expressed threshold, not a measured internal reasoning process.",
    ),
    "C30": (
        "Personal-versus-group reading with altered spelling",
        "Both systems call the statement about women a personal complaint rather than a group attack. The spelling s1ck appears in the input, but the rationales indicate they understood its meaning.",
        "Do not attribute the miss to obfuscation alone: the saved evidence supports a group-target/threshold disagreement, and the spelling effect is not isolated.",
    ),
    "C31": (
        "Warning about a threat versus making that threat",
        "Full retrieval identifies the sentence as a warning about consequences. No retrieval extracts the referenced threat and classifies the warning itself as high-severity hate.",
        "This is a clear attribution/stance fix against the diagnostic label. The experiment does not validate the sentence’s legal claim.",
    ),
    "C32": (
        "Rebuke of an aggressor assigned to the protected group",
        "Full retrieval treats a rebuke directed at someone insulting immigrants as if it insulted immigrants. No retrieval distinguishes the addressee from the group and matches non-hate gold.",
        "This stable regression shows the same stance/quotation problem in the opposite direction to C31; detailed context causes remain for Part 6.",
    ),
    "C33": (
        "Condemned slur treated as speaker endorsement",
        "Both systems classify the reported racist phrase as hate even though the complete sentence explicitly condemns its speaker. Their outputs attach race and high severity.",
        "The presence of a slur does not by itself identify the current speaker’s stance. This is a persistent error under the non-hate diagnostic label.",
    ),
    "C34": (
        "Rhetorical question read as endorsement or challenge",
        "Full retrieval matches hateful gold for the question devaluing immigrants’ lives. No retrieval reads it as challenging the harmful belief and therefore withholds hate and fine labels.",
        "The example shows that question form alone cannot determine stance. It does not establish the correctness of every rhetorical-question interpretation.",
    ),
    "C35": (
        "Broad group disgust dismissed as personal feeling",
        "Full retrieval treats expressed disgust toward women as insufficiently intense/specific for hate. No retrieval matches the hateful diagnostic gate and supplies gender labels.",
        "The stricter reading loses the downstream gender field. This is a stable new miss even though full retrieval has the better aggregate HateCheck F1.",
    ),
    "C36": (
        "Recognized obfuscated insult, different hate threshold",
        "Both systems recognize the spaced-out gendered slur but classify it as a personal insult rather than hate. The diagnostic labels it hateful; the fine fields stay empty.",
        "The miss cannot simply be attributed to failure to decode the spacing, since the rationales name the underlying insult. Legal output, where present, remains independent of hate.",
    ),
}


def run():
    e.r.verify(e.c.read(e.BASE / "analysis_complete.json")["outputs"])
    cases = e.c.read(e.BASE / "cases.json")
    e.require(
        set(NOTES) == {case["case_id"] for case in cases},
        "Manual review roster incomplete",
    )
    records = []
    for case in cases:
        theme, reading, qualification = NOTES[case["case_id"]]
        records.append(
            dict(
                case_id=case["case_id"],
                key=case["key"],
                theme=theme,
                reading=reading,
                qualification=qualification,
                gold_changed=False,
                all_nine_delivered_decisions_reviewed=True,
                rationale_scope="First outer repeat of full and the named reference, as saved; never a faithful-reasoning claim.",
                source_case_sha256=e.c.signature(case),
            )
        )
    e.save(
        e.BASE / "manual_case_review.json",
        dict(
            status="all_36_fixed_examples_reviewed",
            records=records,
            limits="Outcome-stratified illustrative selection, not a prevalence sample. No relabelling, new annotations by participants, legal validation or retrieval-causal inference.",
            sources=e.bindings(
                [
                    Path(__file__),
                    e.BASE / "cases.json",
                    e.BASE / "case_selection.json",
                    e.ROOT / "config/taxonomy.yaml",
                ]
            ),
        ),
    )


if __name__ == "__main__":
    with e.r.original.offline():
        run()
