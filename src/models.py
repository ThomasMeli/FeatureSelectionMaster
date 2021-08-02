
class ModelManager:

    def _check_models(self, models):
        # todo: incorrect - causes crash with ensemble estimators
        model = models

        if hasattr(models, "fit"):
            # The user set models as a particular model instead of a list.
            return models
        else:
            if len(models) > 1:
                return models[0]
            else:
                print("error setting model... Setting to Ridge()")
                return Ridge()