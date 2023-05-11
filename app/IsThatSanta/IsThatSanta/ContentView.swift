//
//  ContentView.swift
//  IsThatSanta
//
//  Created by Ian Maloney on 5/7/23.
//

import SwiftUI
import Photos
import TensorFlowLite

struct ContentView: View {
    @State private var showImagePicker = false
    @State private var capturedImage: UIImage?
    @State private var resultText = "Result: "
    
    let modelFileName = "run_2"
    let imgHeight = 256
    let imgWidth = 256
    
    var body: some View {
        VStack {
            
            Text("Is that Santa")
                    .font(.custom("Chalkboard SE", size: 24))
                    .padding(.top)
            Image("santa")
                .resizable()
                .scaledToFit()
                .padding()
            Button(action: {
                self.showImagePicker = true
            }) {
                Text("Take Picture")
                    .font(.custom("Chalkboard SE", size: 24))
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .sheet(isPresented: $showImagePicker, onDismiss: runModel) {
                ImagePicker(image: self.$capturedImage, sourceType: .camera)
            }

            Text(resultText)
                .font(.custom("Chalkboard SE", size: 24))
                .padding(.top)
        }
    }
    
    func runModel() {
        guard let image = capturedImage else { return }

        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: image)
        }, completionHandler: { success, error in
            if let error = error {
                print("Error saving image: \(error)")
            } else {
                print("Image saved successfully")
            }
        })
        let santaClassifier = SantaClassifier(modelFileName: self.modelFileName, inputWidth: self.imgWidth, inputHeight: self.imgHeight)
        if let isSanta = santaClassifier?.classify(image) {
                resultText = isSanta ? "That is Santa" : "That is not Santa"
        } else {
            resultText = "Sorry, I don't know!"
        }
    }
    
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView().statusBarHidden(false)
    }
}

