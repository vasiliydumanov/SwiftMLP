//
//  Model.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 2/28/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

public typealias SerializedModelData = [(name: String, data: SerializedLayerData)]

public final class Model {
    private let _layers: [Layer]
    private var _layerWithParams: [LayerWithParameters] {
        return _layers.compactMap { $0 as? LayerWithParameters }
    }
    
    public private(set) var isCompiled = false
    private var _loss: Loss?
    
    public init(_ layers: [Layer]) {
        _layers = layers
    }
    
    public func compile(loss: Loss) {
        isCompiled = true
        _loss = loss
    }
    
    public func train(x: matrix, y: matrix, optimizer: Optimizer, nEpochs: Int = 1000, batchSize: Int = 32,
                      validationPct: Double = 0.0, metrics: [Metric] = []) {
        guard isCompiled else {
            fatalError("Model must be compiled before training.")
        }
        guard let loss = _loss else { return }
        
        let valSize = Int(round(x.rows * validationPct))
        var xVal: matrix? = nil
        var yVal: matrix? = nil
        var xTrain: matrix
        var yTrain: matrix
        if validationPct > 0 {
            xVal = x[0..<valSize, 0..<x.columns]
            xTrain = x[valSize..<x.rows, 0..<x.columns]
            yVal = y[0..<valSize, 0..<y.columns]
            yTrain = x[valSize..<y.rows, 0..<y.columns]
        } else {
            xTrain = x
            yTrain = y
        }
        
        let shuffledIdx = randperm(xTrain.rows)
        xTrain = xTrain[shuffledIdx, arange(xTrain.columns)]
        yTrain = yTrain[shuffledIdx, arange(yTrain.columns)]

        let batchIds = Array(0..<xTrain.rows).chunked(into: batchSize)
        
        for epoch in 0..<nEpochs {
            for (i, batch) in batchIds.enumerated() {
                let xBatch = xTrain[vector(batch), arange(xTrain.columns)]
                let yBatch = yTrain[vector(batch), arange(yTrain.columns)]
                
                let yBatchPred = _layers.reduce(xBatch, { input, layer in layer.forward(input) })
                
                let backpropLrs: [Layer]
                let initGrad: matrix
                if
                    _layers.last is Softmax,
                    loss is SoftmaxCrossentropy
                {
                    backpropLrs = Array(_layers.dropLast())
                    initGrad = yBatchPred - yBatch
                } else {
                    backpropLrs = _layers
                    initGrad = loss.backprop(y: yBatch, yPred: yBatchPred)
                }
                
                _ = backpropLrs.reversed().reduce(initGrad, { grad, layer in layer.backprop(grad) })
                
                if i == batchIds.count - 1 {
                    var logStr: String = "Epoch: \(epoch + 1)/\(nEpochs): "
                    let trainLossVal = mean(loss.evaluate(y: yBatch, yPred: yBatchPred))
                    logStr += "train loss = \(trainLossVal)"
                    logStr += metrics.map { m in
                        "train \(m.name) = \(m.evaluate(y: yBatch, yPred: yBatchPred))"
                    }.joined(separator: ", ")
                    if let xVal = xVal, let yVal = yVal {
                        let yValPred = _layers.reduce(xVal, { input, layer in layer.forward(input) })
                        let valLossVal = mean(loss.evaluate(y: yVal, yPred: yValPred))
                        logStr += "; val loss = \(valLossVal)"
                        logStr += metrics.map { m in
                            "val \(m.evaluate(y: yVal, yPred: yValPred))"
                        }.joined(separator: ", ")
                    }
                }
            }
        }
        
    }
    
    public func predict(_ x: matrix) -> matrix {
        return _layers.reduce(x, { input, layer in layer.forward(input) })
    }
    
    public func serialize() -> SerializedModelData {
        return _layerWithParams
            .enumerated()
            .map { idx, lr in
                ("\(NSStringFromClass(type(of: lr)))_\(idx + 1)", lr.encode())
            }
    }
    
    public func restore(from data: SerializedModelData) {
        for entry in data {
            let nameComponents = entry.name.split(separator: "_").map(String.init)
            let lrCls = nameComponents[0]
            guard let lrId = Int(nameComponents[1]) else {
                fatalError("Unable to parse layer data.")
            }
            guard let lr = _layerWithParams
                .enumerated()
                .first(where: { idx, lr in
                    NSStringFromClass(type(of: lr)) == lrCls && idx == lrId
                })?.element
            else {
                fatalError("Unable to find layer with parsed class and index.")
            }
            lr.decode(entry.data)
        }
    }
}
