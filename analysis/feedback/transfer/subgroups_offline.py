from analysis.feedback.transfer import archive as a

if __name__ == "__main__":
    with a.offline():
        from analysis.feedback.transfer.subgroups import calculate

        calculate()
