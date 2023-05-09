import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;


/***
 * 
 * This project is used to classify rgb images as we classified in model .
 */
late final interpreter;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // try {
    interpreter = await tfl.Interpreter.fromAsset('deasesModelVersion8.tflite');
    print('Interpreting model done successfully ✔');
  // } catch (e) {
  //   print('Error while interpreting model !!');
  // }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('TFLite Example'),
        ),
        body: Center(
          child: ElevatedButton(
            child: const Text('Pick Image'),
            onPressed: () async {
              final image =
                  await ImagePicker().getImage(source: ImageSource.gallery);
              if (image == null) return;
              final result = await predictImage(File(image.path));
              print(result);
            },
          ),
        ),
      ),
    );
  }

Future<String> predictImage(File imageFile) async {
  try {
    // Load the image and resize it to the expected input size
    img.Image originalImage = img.decodeImage(await imageFile.readAsBytes())!;
    int inputSize = interpreter.getInputTensor(0).shape[1];
    img.Image resizedImage = img.copyResize(originalImage, width: inputSize, height: inputSize);

    // Normalize the image pixels (assuming pixel values in range [0, 255])
    var imageBytes = resizedImage.getBytes();
    var normalizedPixels = imageBytes.map((pixelValue) => pixelValue / 255.0).toList();

    // Create a 4-dimensional input tensor (assuming RGB image)
    var input = Float32List(inputSize * inputSize * 3).reshape([1, inputSize, inputSize, 3]);
    for (int i = 0; i < inputSize * inputSize; ++i) {
      input[0][i ~/ inputSize][i % inputSize] = [
        normalizedPixels[i * 3],
        normalizedPixels[i * 3 + 1],
        normalizedPixels[i * 3 + 2]
      ];
    }

    // Run the interpreter
    final inputShape = interpreter.getInputTensor(0).shape;
    if (inputShape[0] != 1) {
      throw Exception('Invalid input batch size ${inputShape[0]}, expected 1.');
    }
    final outputShape = interpreter.getOutputTensor(0).shape;
    final output = List.filled(outputShape[0], 0).reshape(outputShape);
    interpreter.run(input, output);
    return output.toString();
  } catch (e) {
    print('Error while predicting image: $e');
    return 'Error';
  }
}
}