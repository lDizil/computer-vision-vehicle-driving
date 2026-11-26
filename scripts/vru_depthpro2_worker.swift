import CoreML
import Darwin
import Foundation

private let width = 640
private let height = 360
private let slots = 6
private let rgbBytes = width * height * 3
private let request = Data("DPQ2".utf8)
private let response = Data("DPR2".utf8)
private let errorResponse = Data("DPE2".utf8)

private enum WorkerError: Error, LocalizedError {
    case invalidArguments
    case invalidMessage
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .invalidArguments:
            return "Expected a compiled model, or --compile <.mlpackage> <new .mlmodelc>."
        case .invalidMessage:
            return "Invalid RGB request or interrupted frame."
        case .invalidOutput:
            return "Depth Pro 2 returned an unexpected inverse_depth shape or type."
        }
    }
}

@available(macOS 15.0, *)
private func main(_ arguments: [String]) throws {
    if arguments.count == 3 && arguments[0] == "--compile" {
        let src = URL(fileURLWithPath: arguments[1], isDirectory: true)
        let dst = URL(fileURLWithPath: arguments[2], isDirectory: true)
        guard src.pathExtension == "mlpackage", dst.pathExtension == "mlmodelc",
              !FileManager.default.fileExists(atPath: dst.path) else {
            throw WorkerError.invalidArguments
        }
        let compiled = try MLModel.compileModel(at: src)
        try FileManager.default.copyItem(at: compiled, to: dst)
        return
    }
    guard arguments.count == 1 else { throw WorkerError.invalidArguments }
    let url = URL(fileURLWithPath: arguments[0], isDirectory: true)
    guard url.pathExtension == "mlmodelc" else { throw WorkerError.invalidArguments }

    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndGPU
    config.functionName = "res_640x360"
    let model = try MLModel(contentsOf: url, configuration: config)
    let tensor = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)], dataType: .float16)
    let provider = try MLDictionaryFeatureProvider(dictionary: ["frame": MLFeatureValue(multiArray: tensor)])
    var states = [MLState?](repeating: nil, count: slots)
    while let header = try readExactly(8, eofAllowed: true) {
        let slot = Int(header[4])
        let flag = header[5]
        guard header.prefix(4) == request, slot < slots, flag <= 1,
              header[6] == 0, header[7] == 0, let rgb = try readExactly(rgbBytes) else {
            throw WorkerError.invalidMessage
        }
        try autoreleasepool {
            if flag == 1 || states[slot] == nil {
                states[slot] = model.makeState()
            }
            guard let state = states[slot] else { throw WorkerError.invalidMessage }
            fill(tensor, from: rgb)
            let output = try model.prediction(from: provider, using: state)
            guard let inverse = output.featureValue(for: "inverse_depth")?.multiArrayValue else {
                throw WorkerError.invalidOutput
            }
            try writeReply(response, payload: inverseDepthBytes(inverse))
        }
    }
}

private func readExactly(_ count: Int, eofAllowed: Bool = false) throws -> Data? {
    var data = Data(capacity: count)
    while data.count < count {
        guard let part = try FileHandle.standardInput.read(upToCount: count - data.count), !part.isEmpty else {
            if data.isEmpty && eofAllowed { return nil }
            throw WorkerError.invalidMessage
        }
        data.append(part)
    }
    return data
}

private let rgbScale = (0..<256).map { Float16(Float($0) / 255) }

private func fill(_ tensor: MLMultiArray, from rgb: Data) {
    tensor.withUnsafeMutableBufferPointer(ofType: Float16.self) { dst, strides in
        rgb.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            for channel in 0..<3 {
                for row in 0..<height {
                    let source = row * width * 3 + channel
                    let target = channel * strides[1] + row * strides[2]
                    for col in 0..<width {
                        dst[target + col * strides[3]] = rgbScale[Int(src[source + col * 3])]
                    }
                }
            }
        }
    }
}

private func inverseDepthBytes(_ output: MLMultiArray) throws -> Data {
    let shape = output.shape.map { $0.intValue }
    let strides = output.strides.map { $0.intValue }
    guard shape == [1, height, width], strides.count == 3,
          strides[1] > 0, strides[2] > 0 else { throw WorkerError.invalidOutput }
    switch output.dataType {
    case .float16:
        return try pack(output, as: Float16.self, strides: strides)
    case .float32:
        return try pack(output, as: Float.self, strides: strides)
    default:
        throw WorkerError.invalidOutput
    }
}

private func pack<Scalar: BinaryFloatingPoint & MLShapedArrayScalar>(
    _ output: MLMultiArray, as scalar: Scalar.Type, strides: [Int]
) throws -> Data {
    let rowStride = strides[1], columnStride = strides[2]
    let end = (height - 1) * rowStride + (width - 1) * columnStride
    return try output.withUnsafeBufferPointer(ofType: scalar) { values in
        guard end < values.count else { throw WorkerError.invalidOutput }
        var bits = [UInt32](repeating: 0, count: width * height)
        for row in 0..<height {
            for column in 0..<width {
                bits[row * width + column] = Float(values[row * rowStride + column * columnStride]).bitPattern.littleEndian
            }
        }
        return bits.withUnsafeBytes { Data($0) }
    }
}

private func writeReply(_ magic: Data, payload: Data) throws {
    var header = magic
    var size = UInt32(payload.count).littleEndian
    withUnsafeBytes(of: &size) { header.append(contentsOf: $0) }
    try FileHandle.standardOutput.write(contentsOf: header)
    try FileHandle.standardOutput.write(contentsOf: payload)
}

do {
    guard #available(macOS 15.0, *) else {
        throw NSError(domain: "DepthPro2", code: 1, userInfo: [NSLocalizedDescriptionKey: "macOS 15 or later required."])
    }
    try main(Array(CommandLine.arguments.dropFirst()))
} catch {
    let message = Data(error.localizedDescription.utf8.prefix(4096))
    if CommandLine.arguments.dropFirst().first != "--compile" {
        try? writeReply(errorResponse, payload: message)
    }
    try? FileHandle.standardError.write(contentsOf: message + Data([10]))
    exit(1)
}
