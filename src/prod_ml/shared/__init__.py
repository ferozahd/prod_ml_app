'''
src/prod_ml/shared/
├── __init__.py
├── embeddings/            # Enterprise-wide user/item embedding lookups
│   └── user_embeddings.py
├── feature_store/         # Enterprise Feature Store connection pool (Feast/Hopsworks)
│   └── client.py
└── database/              # Relational database connections & ORM sessions
    └── session.py
'''