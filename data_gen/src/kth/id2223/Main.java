package kth.id2223;

import nl.captcha.Captcha;
import nl.captcha.backgrounds.FlatColorBackgroundProducer;
import nl.captcha.gimpy.*;

import javax.swing.*;
import java.awt.*;

public class Main extends JPanel {
    public void paint(Graphics g) {
        Captcha captcha = new Captcha.Builder(200, 100)
                .addText(new DefaultWordRenderer1())
                .gimp(new BlockGimpyRenderer())
                .addBackground(new FlatColorBackgroundProducer(Color.WHITE))
//                .addNoise(new CurvedLineNoiseProducer(Color.blue, 2))
                .build();

        Image img = captcha.getImage();
        g.drawImage(img, 20, 20, this);
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame();
        frame.getContentPane().add(new Main());

        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(200, 200);
        frame.setVisible(true);
    }
}
