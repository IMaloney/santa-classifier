//
//  SantaClassifier.swift
//  IsThatSanta
//
//  Created by Ian Maloney on 5/7/23.
//
import TensorFlowLite
import UIKit

extension UIImage {
    func resize(to newSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
        self.draw(in: CGRect(origin: .zero, size: newSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
}

struct SantaClassifier {
    let interpreter: Interpreter
    let inputWidth: Int
    let inputHeight: Int
    
    init?(modelFileName: String, inputWidth: Int, inputHeight: Int) {
        guard let modelPath = Bundle.main.path(forResource: modelFileName, ofType: "tflite") else {
            print("Failed to find the model file.")
            return nil
        }
        
        do {
            self.interpreter = try Interpreter(modelPath: modelPath)
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        }
    
    func preprocessImage(_ image: UIImage) -> Data? {
        guard let resizedImage = image.resize(to: CGSize(width: inputWidth, height: inputHeight)) else {
            print("Failed to resize the image.")
            return nil
        }
        return resizedImage.jpegData(compressionQuality: 1.0)
    }
    
    func classify(_ image: UIImage) -> Bool? {
        guard let inputData: Data = self.preprocessImage(image) else {
            return nil
        }
       do {
           try self.interpreter.allocateTensors()
           
           try self.interpreter.copy(inputData, toInputAt: 0)
           try self.interpreter.invoke()
           let outputTensor = try self.interpreter.output(at: 0)
           let outputSize = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
           let outputData = UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize)
           _ = outputTensor.data.copyBytes(to: outputData)
           let outputArray = Array(outputData)
    
           let isSanta = outputArray[0] < outputArray[1]

           outputData.deallocate()
           
           return isSanta
       } catch let error {
           print("Failed to run inference with error: \(error.localizedDescription)")
           return nil
       }
   }
}
