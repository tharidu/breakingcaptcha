package kth.id2223;

import nl.captcha.Captcha;
import nl.captcha.backgrounds.FlatColorBackgroundProducer;
import nl.captcha.gimpy.*;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.nio.charset.StandardCharsets;

public class Main extends JPanel {
    public void paint(Graphics g) {
        Captcha captcha = new Captcha.Builder(216, 128)
                .addText(new DefaultWordRenderer1())
                .gimp(new BlockGimpyRenderer())
                .addBackground(new FlatColorBackgroundProducer(Color.WHITE))
//                .addNoise(new CurvedLineNoiseProducer(Color.blue, 2))
                .build();

        Image img = captcha.getImage();
        g.drawImage(img, 20, 20, this);
    }

    public static void main(String[] args) throws IOException {
//        JFrame frame = new JFrame();
//        frame.getContentPane().add(new Main());
//
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setSize(300, 300);
//        frame.setVisible(true);

        generateDatasets(1000);
    }

    public static void generateDatasets(long datasetSize) throws IOException {
        OutputStream f = new FileOutputStream("data");

        for (int i = 0; i < datasetSize; i++) {
            Captcha captcha = new Captcha.Builder(216, 128)
                    .addText(new DefaultWordRenderer1())
                    .gimp(new BlockGimpyRenderer())
                    .addBackground(new FlatColorBackgroundProducer(Color.WHITE))
                    .build();

            byte[] b = captcha.getAnswer().getBytes(StandardCharsets.US_ASCII);
            byte[] pixels = ((DataBufferByte) captcha.getImage().getRaster().getDataBuffer()).getData();

            f.write(b);
            f.write(pixels);
//            for (int i1 = 0; i1 < pixels.length; i1++) {
//
//            }
//
//            for (byte sb :
//                    pixels) {
//                System.out.println(Byte.toUnsignedInt(sb));
//            }

        }
        f.flush();
        f.close();
    }
}
