def response_fixture(framework="source"):
    form = "synthetic-" + framework
    ids = ["R008", "R023", "R025", "R031", "R039", "R051"] + [
        f"F{i:03}" for i in range(18)
    ]
    mapping = dict(
        form_key="synthetic",
        framework=framework,
        fields={},
        texts={pid: pid for pid in ids},
    )
    questions, responses = [], []
    labels = [
        "explicit",
        "incitement",
        "inferiority",
        "irony",
        "stereotyping",
        "threatening",
        "grievance",
    ]

    def field(key, title, options=None, answer=None, multiple=False):
        number = len(questions)
        group, qid = f"group-{number}", f"question-{number}"
        value = dict(
            group_uuid=group,
            title=title,
            type="CHECKBOXES" if multiple else "MULTIPLE_CHOICE_OPTION",
        )
        if options is not None:
            value["option_uuids"] = {
                label: f"option-{number}-{i}" for i, label in enumerate(options)
            }
            answer = [value["option_uuids"][label] for label in (answer or [])]
        mapping["fields"][key] = value
        questions.append(
            dict(
                id=qid,
                formId=form,
                isDeleted=False,
                title=title,
                fields=[dict(uuid=group, blockGroupUuid=group)],
            )
        )
        responses.append(
            dict(
                questionId=qid,
                formId=form,
                submissionId="synthetic-response",
                answer=answer,
            )
        )

    for pid in ids:
        title = pid + " " + chr(183) + " "
        if framework == "source":
            gate = (
                "Not hate"
                if pid == "R051"
                else "Uncertain" if pid in {"R025", "R031"} else "Implicit hate"
            )
            field(
                pid + "/gate",
                title + "Classification",
                ["Explicit hate", "Implicit hate", "Not hate", "Uncertain"],
                [gate],
            )
            primary = (
                []
                if pid in {"R023", "R025", "R031", "R051"}
                else (
                    ["incitement"]
                    if pid == "R008"
                    else ["grievance"] if pid == "R039" else ["irony"]
                )
            )
            secondary = (
                ["grievance"]
                if pid == "R039"
                else ["No secondary type"] if pid == "R008" else []
            )
            field(
                pid + "/primary",
                title + "Primary implicit type (if applicable)",
                labels,
                primary,
            )
            field(
                pid + "/secondary",
                title + "Secondary implicit type (if applicable)",
                labels + ["No secondary type"],
                secondary,
            )
        else:
            field(
                pid + "/gate",
                title + "Classification",
                ["Yes", "No", "Uncertain"],
                ["Yes"],
            )
            field(pid + "/types", title + "Types", labels, ["irony"], multiple=True)
        field(
            pid + "/flags", title + "Difficulty", ["Context missing"], [], multiple=True
        )
        field(pid + "/guide_help", title + "Guide", ["Show guide"], [], multiple=True)
    field("notes", "Notes", answer="ASTRA_QA_ONLY_SYNTHETIC_DECODER_FIXTURE")
    field("minutes", "Minutes", answer=0)
    field("independence", "Independent", ["Yes", "No"], ["Yes"])
    data = dict(
        hasMore=False,
        questions=questions,
        submissions=[
            dict(
                id="synthetic-response",
                formId=form,
                isCompleted=True,
                responses=responses,
            )
        ],
    )
    return data, mapping, form
