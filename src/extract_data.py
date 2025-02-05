import xml.etree.ElementTree as ET

def extract_relevant_info(file_path):
    """
    Extract 'relevant info' from an XML file:
      1) Root <Statement> (if any).
      2) For each <ProblemStep>:
         - The <Statement> text
         - For each <Hint>: the <Text> content
      Skips <Answer>, <AlternateAnswers>, <RandomVariables>, etc.

    Returns a single string of combined relevant text.
    """
    try:
        # Read file as bytes
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        # Attempt UTF-8 decode, fallback to UTF-16
        try:
            root = ET.fromstring(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            root = ET.fromstring(raw_bytes.decode("utf-16"))

        collected_text = []

        # 1) Root problem statement
        root_statement = root.find("Statement")
        if root_statement is not None and root_statement.text:
            text_str = root_statement.text.strip()
            if text_str:
                collected_text.append(text_str)

        # 2) For each ProblemStep => gather statement & hints
        for step in root.findall(".//ProblemStep"):
            # <Statement> inside the ProblemStep
            step_statement = step.find("Statement")
            if step_statement is not None and step_statement.text:
                s_str = step_statement.text.strip()
                if s_str:
                    collected_text.append(s_str)

            # <Hint> elements: each has a <Text> child
            for hint in step.findall(".//Hint"):
                hint_text_el = hint.find("Text")
                if hint_text_el is not None and hint_text_el.text:
                    h_str = hint_text_el.text.strip()
                    if h_str:
                        collected_text.append(h_str)

        # Combine into one big text block
        return "\n".join(collected_text)

    except Exception as e:
        # Raise a RuntimeError so callers can handle or log
        raise RuntimeError(f"Failed to parse '{file_path}': {str(e)}")
