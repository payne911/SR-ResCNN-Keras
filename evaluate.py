def evaluate(model, trainingTestData, validateTestData):

    # TODO: Now use the TEST dataset to calculate performance
    test_loss, test_acc = model.evaluate(trainingTestData, validateTestData)
    print('Test accuracy:', test_acc)
