//
//  AppDelegate.swift
//  SwiftMLP-Demo
//
//  Created by Vasiliy Dumanov on 2/28/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import UIKit
import swix_ios

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        demo()
        return true
    }

    
    func demo() {
        let fileNames = ["w1", "w2", "x", "y"]
        let shapes: [(Int, Int)] = [(2, 3), (3, 2), (1, 2), (1, 1)]
        var params: [matrix] = []
        for (name, shape) in zip(fileNames, shapes) {
            let filePath = Bundle.main.path(forResource: name, ofType: "csv")!
            let fileContents = try! String(contentsOfFile: filePath, encoding: .utf8)
            let elements = fileContents.split(separator: "\n").map { Double($0)! }
            params.append(
                vector(elements).reshape(shape)
            )
        }
        
        let w1 = params[0]
        let w2 = params[1]
        let x = params[2]
        let y = onehot(params[3].flat, nClasses: 2)
        
        let model = Model([
            Dense(units: 3,
                  weightsInitializer: ConstantInitializer(w1),
                  biasInitializer: ZerosInitializer()),
            Relu(),
            Dense(units: 2,
                  weightsInitializer: ConstantInitializer(w2),
                  biasInitializer: ZerosInitializer()),
            Softmax()
            ])
        model.compile(loss: SoftmaxCrossentropy())
        model.train(x: x, y: y, optimizer: SGDOptimizer())
    }
}

