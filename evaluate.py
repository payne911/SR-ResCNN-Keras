def evaluate(model, X_test, Y_test):

    # TODO: Now use the TEST dataset to calculate performance
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)
