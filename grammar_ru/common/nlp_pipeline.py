# Pipeline that combines two things:
# 1) NlpAnalyzationPipeline
# 2) NlpPreprocessingPipeline

# Scenarios:
# 1) I should be able to send NlpAnalizationPipeline to create dataset to cloud
#   because it can require a lot of resources. It means, that NlpAnalyzationPipeline should
#   be either inherited from FeaturizationJob or be able to return it conviniently.
# 2) I should be able to apply NlpPreprocessingPipeline to apply preprocessing before passing
#   the data to the algorithm.


# An algorithm should receive NlpPipeline containing both analyzation and preprocessing
# in order to be able to transform new data accordingly.
# The problem here is that we probably want to prevent scenarios when you trained model using one
# pipeline and then applying model using other pipeline. It could be prevented if NlpAlgorithm
# contained some methods for model training, but I think that this architecture would not be
# flexible and convenient enough for testing and experimenting with data.
# So the user should be careful when analyzing and applying its results.
