import getml

getml.set_project("loans")

population_train, population_test, order, trans, meta = getml.datasets.load_loans()

schema = getml.data.StarSchema(
    train=population_train,
    test=population_test,
    alias="population",
)

schema.join(
    trans,
    on="account_id",
    time_stamps=("date_loan", "date"),
)

schema.join(
    order,
    on="account_id",
)

schema.join(
    meta,
    on="account_id",
)

relmt = getml.feature_learning.RelMT(
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
)

xgboost = getml.predictors.XGBoostClassifier()

pipe = getml.pipeline.Pipeline(
    data_model=schema.data_model,
    feature_learners=relmt,
    predictors=xgboost,
)

pipe.fit(schema.train)