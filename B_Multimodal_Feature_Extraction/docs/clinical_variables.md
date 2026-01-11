# Clinical Variables

Clinical variables provide complementary non-imaging information and are
incorporated as an independent modality.

---

## Examples of Clinical Variables

Clinical features may include:

- Demographic information (e.g., age, sex)
- Laboratory indicators
- Tumor-related clinical findings
- Staging or grading information (if available)

---

## Preprocessing

- Categorical variables should be encoded numerically.
- Continuous variables may be standardized before model training.
- Missing values should be handled consistently (e.g., imputation).

---

## Output Format

- CSV file with one row per case
- `ID` column for alignment
- Remaining columns represent clinical variables
