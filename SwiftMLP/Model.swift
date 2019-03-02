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
        
        let shuffledIdx = randperm(x.rows)
        let shuffledX = x[shuffledIdx, arange(x.columns)]
        let shuffledY = y[shuffledIdx, arange(y.columns)]
        
        let valSize = Int(round(x.rows * validationPct))
        var xVal: matrix? = nil
        var yVal: matrix? = nil
        let xTrain: matrix
        let yTrain: matrix
        if validationPct > 0 {
            xVal = shuffledX[0..<valSize, 0..<x.columns]
            xTrain = shuffledX[valSize..<x.rows, 0..<x.columns]
            yVal = shuffledY[0..<valSize, 0..<y.columns]
            yTrain = shuffledY[valSize..<y.rows, 0..<y.columns]
        } else {
            xTrain = x
            yTrain = y
        }
        
        let batchIds = Array(0..<xTrain.rows).chunked(minSize: batchSize)
        let isSoftmaxAndCrossentropy = _layers.last is Softmax && loss is SoftmaxCrossentropy
        let backpropLrs = isSoftmaxAndCrossentropy ? Array(_layers.dropLast()) : _layers
        
        _layerWithParams.forEach { lr in lr.resetStates() }
        for epoch in 0..<nEpochs {
            for (i, batch) in batchIds.enumerated() {
                let xBatch = xTrain[vector(batch), arange(xTrain.columns)]
                let yBatch = yTrain[vector(batch), arange(yTrain.columns)]
                
                let xBatchExploded = vexplode(xBatch).map { reshape($0, shape: (1, $0.n)) }
                let yBatchExploded = vexplode(yBatch).map { reshape($0, shape: (1, $0.n)) }
                
                var ySinglePreds: [vector] = []
                _layerWithParams.forEach { lr in lr.resetGradients() }
                for (xSingle, ySingle) in zip(xBatchExploded, yBatchExploded) {
                    let ySinglePred = _layers.reduce(xSingle, { input, layer in layer.forward(input) })
                    ySinglePreds.append(ySinglePred.flat)
                    let initGrad: matrix
                    if isSoftmaxAndCrossentropy {
                        initGrad = ySinglePred - ySingle
                    } else {
                        initGrad = loss.backprop(y: ySingle, yPred: ySinglePred)
                    }
                    _ = backpropLrs.reversed().reduce(initGrad, { grad, layer in layer.backprop(grad) })
                }
                _layerWithParams.forEach { lr in
                    optimizer.optimizeGradients(for: lr, epoch: epoch + 1)
                }
                
                if i == batchIds.count - 1 {
                    let yBatchPred = vstack(ySinglePreds)
                    var logStr: String = "Epoch: \(epoch + 1)/\(nEpochs): "
                    var trainStrs: [String] = []
                    let trainLossVal = mean(loss.evaluate(y: yBatch, yPred: yBatchPred))
                    trainStrs.append("train loss = \(trainLossVal)")
                    trainStrs += metrics.map { m in
                        "train \(m.name) = \(m.evaluate(y: yBatch, yPred: yBatchPred))"
                    }
                    logStr += trainStrs.joined(separator: ", ")
                    if let xVal = xVal, let yVal = yVal {
                        let yValPred = _layers.reduce(xVal, { input, layer in layer.forward(input) })
                        var valStrs: [String] = []
                        let valLossVal = mean(loss.evaluate(y: yVal, yPred: yValPred))
                        valStrs.append("; val loss = \(valLossVal)")
                        valStrs += metrics.map { m in
                            "val \(m.name) = \(m.evaluate(y: yVal, yPred: yValPred))"
                        }
                        logStr += valStrs.joined(separator: ", ")
                    }
                    print(logStr)
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
    
    public func save(to modelDir: String) throws {
        let fm = FileManager.default
        try? fm.removeItem(atPath: modelDir)
        try serialize().forEach { lrName, lrData in
            let lrDirPath = (modelDir as NSString).appendingPathComponent(lrName)
            if !fm.fileExists(atPath: lrDirPath) {
                try fm.createDirectory(atPath: lrDirPath, withIntermediateDirectories: true, attributes: nil)
            }
            lrData.forEach { paramName, paramData in
                let entryFilePath = (lrDirPath as NSString).appendingPathComponent(lrName)
                write_binary(paramData, filename: entryFilePath)
            }
        }
    }
    
    public func restore(from modelDir: String) throws -> Bool {
        let fm = FileManager.default
        guard let lrDirs = try? fm.contentsOfDirectory(atPath: modelDir) else { return false }
        try lrDirs.forEach { lrDir in
            let lrParamFiles = try fm.contentsOfDirectory(atPath: (modelDir as NSString).appendingPathComponent(lrDir))
            var lrData: SerializedLayerData = [:]
            lrParamFiles.forEach { lrParamFile in
                let paramName = lrParamFile
                let paramData: matrix = read_binary(lrParamFile)
                lrData[paramName] = paramData
            }
        }
        return true
    }
}
