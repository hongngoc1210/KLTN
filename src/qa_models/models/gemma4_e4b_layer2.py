from qa_models.tasks.layer2.generative_adapter import GenerativeLayer2Adapter


class Gemma4E4BLayer2ZeroShotAdapter(GenerativeLayer2Adapter):
    """Gemma-4-E4B-it layer-2 JSON prompting."""

    def chat_template_kwargs(self):
        return {}
