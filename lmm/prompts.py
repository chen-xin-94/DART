def get_prompts(target, secondary_target=None):

    def preprocessing(target):
        if isinstance(target, list) and len(target) > 1:
            target = "s and ".join(target)
        elif isinstance(target, list) and len(target) == 1:
            target = target[0]
        elif isinstance(target, str):
            pass
        elif target is None:
            pass
        else:
            raise ValueError("Argument should be a list of strings or a string.")
        return target

    target = preprocessing(target)
    secondary_target = preprocessing(secondary_target)
    if secondary_target:
        task_txt = f"""In this image, the target object for bounding box annotation is {target}. There may also be {secondary_target}. Each of existing {target}s and {secondary_target}s should be annotated by a bounding box drawn as a colored rectangle. The goal is to accurately localize all {target}s and {secondary_target}s using these bounding boxes. Your task is to evaluate whether all bounding boxes are correct.
"""
    else:
        task_txt = f"""In this image, the target object for bounding box annotation is {target}. Each of existing {target}s should be annotated by a bounding box drawn as a colored rectangle. The goal is to accurately localize all {target}s using these bounding boxes. Your task is to evaluate whether all bounding boxes are correct.
"""
    if secondary_target:
        relax_txt = f"""Before finalizing your evaluation, please consider the following suggestions:
1. For question 1 (Precision), if an object is occluded, the bounding box should be inferred based on a reasonable estimation of the object's size.
2. For question 2 (Recall),  not all target objects should be in the image. It's acceptable to only have one {target} but no {secondary_target} in the image.
3. For question 3 (Fit), don't be too harsh when bounding box edges just slightly cut off the object or just enclose a little bit of the outside area.
"""
    else:
        relax_txt = f"""Before finalizing your evaluation, please consider the following suggestions:
1. For question 1 (Precision),  if an object is occluded, the bounding box should be inferred based on a reasonable estimation of the object's size.
2. For question 2 (Recall),  it's fairly normal to have only one object in the dataset.
3. For question 3 (Fit), don't be too harsh when bounding box edges just slightly cut off the object or just enclose a little bit of the outside area.
"""

    prompt_templates = {
        "system": "You are an AI bounding box annotation evaluator. Your task is to evaluate the correctness of bounding box annotations for given images and target objects. The bounding boxes are directly drawn as colored rectangles on top of the image. The class label is shown at the top-left corner of the corresponding bounding box. You will evaluate each bounding box annotation based on three criteria: precision, recall, and fit.",
        "task": task_txt,
        "questions": f"""Correctness should be assessed in terms of precision, recall, and fit.
Specifically, consider the following questions before making your judgement:
1. Does each bounding box perfectly enclose one single target object? 
2. Are all target objects localized by a bounding box?
3. Is each bounding box neither too loose nor too tight?
""",
        "output": f"""Please provide your evaluation in the following JSON format:
```json
{{"Precision":"Yes/No answer to question 1", "Recall":"Yes/No answer to question 2", "Fit":"Yes/No answer to question 3"}}
```
Please think step-by-step and be sure to provide the correct answers. Very briefly explain yourself before answering the question.
""",
        "be-relax": relax_txt,
        #         "be-rigorous": f"""Please take the following suggestions into account before making your decision. Even though the bounding boxes are rectangular and not pixel-perfect, it is crucial to minimize any inclusion of areas outside the target object. Furthermore, the bounding boxes should tightly enclose the target without cutting off any part of it. Ensure that they do not exclude any major parts of the object. Apply strict criteria to ensure precise and accurate labeling. These are just suggestions, and the most important thing is to answer the three questions correctly.
        # """,
        "caveat": f"""Always considering my suggestions before answering questions. But the suggestions are not stric rules. You can also use your own judegment. The most important thing is to answer the three questions correctly.
""",
    }
    return prompt_templates
