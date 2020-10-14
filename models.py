from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras import initializers
from tensorflow.keras.models import Model

shape=(224,224)

def make_model(model, initial_bias, name):
    x = layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    bias_initializer = initializers.Constant(initial_bias)
    x = layers.Dense(1, activation='sigmoid',name='predictions', bias_initializer=bias_initializer)(x)

    xmodel = Model(inputs=model.input, outputs=x, name=name)

    return xmodel


def VGG16(initial_bias):
    model = applications.VGG16(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3))
    
    for layer in model.layers:
        if layer.name.startswith('block5'):
                break
        layer.trainable = False
        
    xmodel = make_model(model, initial_bias, 'VGG16')
    
    return xmodel, applications.vgg16.preprocess_input


def MobileNet(initial_bias):
    model = applications.MobileNet(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3)
    )
    for layer in model.layers:
        layer_name = layer.name.split('_')
        if layer_name[0] == 'conv' and layer_name[2] == '13':
            break
        layer.trainable = False

    xmodel = make_model(model, initial_bias, 'MobileNet')

    return xmodel, applications.mobilenet.preprocess_input


def MobileNetV2(initial_bias):
    model = applications.MobileNetV2(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3)
    )

    for layer in model.layers:
        layer_name = layer.name.split('_')
        if layer_name[0] == 'conv' and layer_name[2] == '16':
            break
        layer.trainable = False

    xmodel = make_model(model, initial_bias, 'MobileNetV2')

    return xmodel, applications.mobilenet_v2.preprocess_input


def ResNet50V2(initial_bias):
    model = applications.ResNet50V2(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3)
    )
    
    for layer in model.layers:
        if layer.name.startswith('conv5'):
            break
        layer.trainable = False
    
    xmodel = make_model(model, initial_bias, 'ResNet50V2')
    
    return xmodel, applications.resnet_v2.preprocess_input


def ResNet152V2(initial_bias):
    model = applications.ResNet152V2(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3)
    )

    for layer in model.layers:
        if layer.name.startswith('conv5'):
            break
        layer.trainable = False
    
    xmodel = make_model(model, initial_bias, 'ResNet152V2')
    
    return xmodel, applications.resnet_v2.preprocess_input    


def Xception(initial_bias):
    model = applications.Xception(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3)
    )

    for layer in model.layers:
        if layer.name.startswith('block14'):
            break
        layer.trainable = False
    
    xmodel = make_model(model, initial_bias, 'Xception')
    
    return xmodel, applications.xception.preprocess_input  


def InceptionResNetV2(initial_bias):
    model = applications.InceptionResNetV2(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3)
    )

    for layer in model.layers:
        if layer.name.startswith('block8'):
            break
        layer.trainable = False
    
    xmodel = make_model(model, initial_bias, 'InceptionResNetV2')
    
    return xmodel, applications.inception_resnet_v2.preprocess_input


def InceptionV3(initial_bias):
    model = applications.InceptionV3(
                weights="imagenet", 
                include_top= False,
                input_shape=(shape[0], shape[1],3)
    )

    for layer in model.layers:
        if layer.name.startswith('conv2d_89'):
            break
        layer.trainable = False
    
    xmodel = make_model(model, initial_bias, 'InceptionV3')

    return xmodel, applications.inception_v3.preprocess_input

models_fn = [
    VGG16,
    MobileNet,
    MobileNetV2,
    ResNet50V2,
    ResNet152V2,
#    Xception,
    InceptionResNetV2,
    InceptionV3
]

def preload():
    for model_fn in models_fn:
        model_fn(0.1)

if __name__ == '__main__':
    preload()
