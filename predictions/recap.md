# GLOBAL CONFIGURATION
NUMBER_OF_RANDOM_SPLITS = 10
TEST_SIZE = 0.35


# Config 1
- ohe su tutto
- remove BsmtQual_Po, MSSubClass_150, MSZoning_C (all)
- drop Utilities
- introduce HasStreet

> DEV ERROR ~ Stats over 10 random splits with 0.35 test
> mean: 0.11399776058678417
> variance: 5.177543602337581e-05
>stdev: 0.007195514993617608


# Config 2
- ohe su tutto
- remove BsmtQual_Po, MSSubClass_150, MSZoning_C (all)
- drop Utilities, Street, HasStreet

> DEV ERROR ~ Stats over 10 random splits with 0.35 test
> mean: 0.11327152866230401
> variance: 3.438688187174615e-05
> stdev: 0.005864032901659586


# Config 3
- ints e ohe ove ecessario
- remove MSSubClass_150, MSZoning_C (all)
- drop Utilities, Street, HasStreet

> DEV ERROR ~ Stats over 10 random splits with 0.35 test
> mean: 0.10947234976385638
> variance: 4.0333508560109784e-05
> stdev: 0.0063508667566017935


# Config 4
- ints e ohe ove ecessario
- remove MSSubClass_150, MSZoning_C (all)
- drop Utilities, Street, HasStreet
- BsmtQual['Po'] = BsmtQual['Fa']

> DEV ERROR ~ Stats over 10 random splits with 0.35 test
> mean: 0.11168406523996574
> variance: 7.096177249945187e-05
> stdev: 0.00842388108293629


# Config 5 (best)
- ints e ohe ove ecessario
- remove MSSubClass_150, MSZoning_C (all)
- drop Utilities, Street, HasStreet
- BsmtQual['Po'] = BsmtQual['TA']

> DEV ERROR ~ Stats over 10 random splits with 0.35 test
> mean: 0.11157626300686145
> variance: 3.431070248297061e-05
> stdev: 0.005857533822605774

> DEV ERROR ~ Stats over 10 random splits with 0.35 test
> mean: 0.11106683389244346
> variance: 4.1615854424412414e-05
> stdev: 0.0064510351436348895

# Config 6
- ints e ohe ove ecessario
- remove MSSubClass_150, MSZoning_C (all)
- drop Utilities, Street, HasStreet
- BsmtQual['Po'] = NONE_VALUE

> DEV ERROR ~ Stats over 10 random splits with 0.35 test
> mean: 0.11185819958360302
> variance: 4.632941178663212e-05
> stdev: 0.006806571221006368


