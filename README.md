# Knee MRI Findings: Radiology Reports and DICOM Analysis

This project explores how knee MRI radiology reports and DICOM images can be processed into structured labels for 12 findings. It combines a rule-based report parser, local Qwen language-model experiments, and a reproducible image pipeline for MRI analysis.

The workflow was developed for testing and experimentation with the RSNA knee MRI dataset. The image model is a separate experimental component: it learns from labeled studies and MRI pixels, while Qwen processes report text.

## Findings analyzed

The task covers ligament and meniscal injuries, degenerative changes, and other findings reported in knee MRI studies:

- **ACL injury** - Abnormality of the anterior cruciate ligament, including sprain, partial or complete tear, or rupture. The parser distinguishes ACL from the posterior cruciate ligament.
- **MCL injury** - Abnormality of the medial collateral ligament, such as sprain or tear.
- **Medial meniscus tear** - Tear or other specified structural injury of the meniscus on the inner side of the knee.
- **Lateral meniscus tear** - Tear or other specified structural injury of the meniscus on the outer side of the knee.
- **Medial compartment osteoarthritis (OA)** - Degenerative cartilage or joint changes in the inner, medial tibiofemoral compartment.
- **Lateral compartment osteoarthritis (OA)** - Degenerative cartilage or joint changes in the outer, lateral tibiofemoral compartment.
- **Patellofemoral osteoarthritis (PF OA)** - Degenerative cartilage changes where the kneecap (patella) meets the thigh bone (femur).
- **Effusion** - Excess joint fluid. The report parser treats explicitly described small or trace effusions as positive findings.
- **Synovitis** - Inflammation or thickening of the synovial lining of the joint.
- **Baker's cyst** - A fluid-filled popliteal cyst behind the knee, also described as a popliteal cyst.
- **Contusion** - Bone bruise, typically described in the context of trauma.
- **Fracture** - An explicitly reported fracture involving the knee bones.

The three OA labels describe different joint compartments; they are degenerative findings rather than acute injuries.

## Turning radiology reports into labels

Reports are linked to studies using `StudyInstanceUID`. The report-labeling workflow combines an English/Spanish rule-based parser with optional local language-model analysis:

1. **Normalize and classify the report.** Text is normalized, its language is detected, and it is split into sentences. The rule-based parser processes English and Spanish reports.
2. **Search for target-specific evidence.** Pattern dictionaries look for finding names, synonyms, tear or degeneration terms, and relevant context for each of the 12 targets.
3. **Check negation and uncertainty.** The parser considers nearby statements such as “intact,” “no tear,” “possible,” or “cannot exclude.” If positive and negative evidence conflict, or a finding is uncertain, the result is kept unresolved rather than forced into a binary label.
4. **Preserve reference labels.** Existing GOLD labels from the source training table take precedence over generated labels. For findings without GOLD labels, clear parser positives or negatives become SILVER labels only when parser confidence is at least 0.85. Unsupported languages, uncertain findings, conflicts, and unresolved cases remain unassigned by the parser.
5. **Optionally analyze unresolved findings with Qwen.** A local Qwen model receives the report and the requested findings, then returns a JSON object with a score for each target. Existing GOLD and accepted SILVER labels remain distinct from these model-generated scores.
6. **Keep provenance with the output.** The merged label table records label value, source, status, confidence or score, and whether it is eligible for training. Missing labels are masked, not interpreted as negative.

This produces auditable structured labels while keeping source labels, rule-based labels, and language-model estimates identifiable. Qwen scores are model-generated estimates, not calibrated clinical probabilities.

## What DICOM means

**DICOM** (Digital Imaging and Communications in Medicine) is the standard format used to store and exchange medical images. A DICOM file typically includes image pixels together with structured metadata, such as study and series identifiers, image position and orientation, acquisition details, and pixel scaling information. A single MRI examination is usually made up of many images grouped into series.

## DICOM image pipeline

The pipeline in `rsna_knee_pipeline.py` processes DICOM studies in the following stages:

1. **Inventory studies and metadata.** Read DICOM headers and group files by study and series. The metadata audit can collect acquisition and image properties without decoding the pixel data.
2. **Select an MRI series.** Rank series using available attributes such as fluid sensitivity, fat suppression, anatomical plane, modality, and slice count. Save the candidate series and the reason for selecting one.
3. **Order the images.** Sort slices by their position in patient space using image position and orientation when those values are available. Fall back to instance number, frame index, and path when necessary.
4. **Decode and preprocess pixels.** Decode single-frame or multi-frame images, apply rescale slope and intercept, handle MONOCHROME1 interpretation, normalize intensities using the 1st and 99th percentiles, and resize with aspect ratio preserved.
5. **Build 2.5D inputs.** Sample up to five evenly distributed locations in the selected series. For each location, stack the preceding, center, and following slices as three image channels. This gives the model nearby-slice context without processing a full 3D volume.
6. **Predict at study level.** Pass the stacks through an ImageNet-initialized ResNet34. Average stack logits to produce one score per finding for the study.

The pipeline also creates audit tables and a small optional image montage to help review series selection and preprocessing.

## Model training and labels

The image model is a multi-label ResNet34 classifier. It can be trained with GOLD labels alone or with optional SILVER and Qwen-derived labels. GOLD labels remain the reference for validation; weak labels are opt-in and receive lower training weights. Missing, uncertain, conflicting, or invalid labels are masked from the loss instead of being treated as negative examples.

Where DICOM patient identifiers are available, train and validation splits are grouped by patient to reduce leakage between studies from the same person. The pipeline records label statistics, split membership, training history, validation results, and selected MRI series.

The Kaggle notebook, `rsna_knee_kaggle.ipynb`, documents the competition input setup and the audit, smoke-test, training, and submission workflow. The training pipeline can also be run from the command line:

```powershell
py -m unittest -v test_rsna_knee_pipeline.py
```

Run a metadata and label audit on an available dataset:

```powershell
py rsna_knee_pipeline.py --input-root . --output-dir . --audit-only --audit-dicom
```

The full RSNA imaging dataset used during development contains approximately **350 GB of DICOM images**. The image data is not included with the source code and must be obtained separately from its official dataset source.

## Local language model

Qwen2.5-7B-Instruct-AWQ was selected because its model weights are freely available to download and it can be run locally, without paying for a hosted LLM API. The report-processing scripts use Transformers to load the local model and request structured JSON scores for the 12 findings. The local workflow supports checkpoints and can resume from previously processed studies.

This Qwen model analyzes report text; it does not analyze the DICOM pixels. MRI pixels are processed separately by the ResNet34 image pipeline. The quantized model weights are approximately 5.2 GB, and local inference requires suitable system memory and hardware. Review the model's license and usage terms before redistribution or deployment.

Originally developed as a hackathon project.
